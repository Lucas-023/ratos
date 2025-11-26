import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
import warnings
import gc

warnings.filterwarnings('ignore')

# =================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =================================================================

# Metadados estáticos para manter
METADATA_COLUMNS_TO_KEEP_RAW = [
    'frame', 'behavior', 'video_id', 'unique_frame_id',
    'lab_id', 'frames_per_second', 'video_duration_sec', 
    'pix_per_cm_approx', 'video_width_pix', 'video_height_pix', 
    'mouse1_strain', 'mouse1_sex', 'mouse2_strain', 'mouse2_sex'
]

# Janelas temporais para cálculo de média/desvio padrão (Rolling Stats)
# Ex: Média de velocidade nos últimos 5, 15 e 30 frames
TEMPORAL_WINDOWS = [5, 15, 30]

# Lags: Olhar para trás X frames (importante para causalidade)
LAG_FEATURES = [1, 5, 10]

# =================================================================
# 2. FUNÇÕES DE CORREÇÃO DE FORMATO (CRÍTICO)
# =================================================================

def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte formato Longo (uma linha por keypoint) para Largo (uma coluna por keypoint).
    Resolve o problema de não achar as colunas de coordenadas.
    """
    required_cols = ['bodypart', 'x', 'y', 'mouse_id']
    # Se não tiver essas colunas, assume que já está largo ou não é tracking padrão
    if not all(col in df.columns for col in required_cols):
        return df

    # Cria identificador único: mouse1, mouse2, etc.
    # Garante que mouse_id seja tratado como string para concatenação
    df['mouse_key'] = 'mouse' + df['mouse_id'].astype(str)
    
    # Pivota a tabela: Transforma linhas em colunas
    pivot_df = df.pivot_table(
        index='video_frame', 
        columns=['mouse_key', 'bodypart'], 
        values=['x', 'y']
    )
    
    # Achata as colunas multinível (ex: ('x', 'mouse1', 'nose') -> 'mouse1_nose_x')
    new_columns = []
    for val_type, mouse, part in pivot_df.columns:
        new_columns.append(f"{mouse}_{part}_{val_type}")
    
    pivot_df.columns = new_columns
    pivot_df = pivot_df.reset_index()
    
    # Padroniza nome da coluna de frame
    pivot_df = pivot_df.rename(columns={'video_frame': 'frame'})
    
    return pivot_df

# =================================================================
# 3. FUNÇÕES AUXILIARES DE CARREGAMENTO
# =================================================================

def load_sequence_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path or not metadata_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(metadata_path)
    except Exception:
        return pd.DataFrame()

def build_metadata_lookup(metadata_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    if metadata_df.empty or 'sequence_id' not in metadata_df.columns:
        return {}
    metadata_df = metadata_df.set_index('sequence_id')
    return {int(idx): row.to_dict() for idx, row in metadata_df.iterrows()}

def load_annotation_file(sequence_id: str, lab_name: str, annotations_root: Optional[Path]) -> pd.DataFrame:
    """Carrega anotações tentando vários caminhos possíveis."""
    candidate_files = []
    if annotations_root and annotations_root.exists():
        candidate_files.append(annotations_root / lab_name / f"{sequence_id}.parquet")
        candidate_files.append(annotations_root / f"{sequence_id}.parquet")

    for candidate in candidate_files:
        if candidate.exists():
            try:
                return pd.read_parquet(candidate, engine='fastparquet')
            except Exception:
                continue
    return pd.DataFrame()

def _aggregate_behaviors(values: pd.Series) -> Any:
    """Junta múltiplos comportamentos no mesmo frame em uma string única."""
    behaviors = set()
    for value in values:
        val_str = str(value).strip()
        if not val_str or val_str.lower() in ['nan', 'none']:
            continue
        # Separa por ponto e vírgula se já vier agrupado
        for sub_val in val_str.split(';'):
            if sub_val.strip():
                behaviors.add(sub_val.strip())
    
    if not behaviors:
        return None
    return ";".join(sorted(behaviors))

def merge_tracking_and_annotations(tracking_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    df = tracking_df.copy()
    if 'frame' not in df.columns:
        df['frame'] = np.arange(len(df), dtype=np.int32)

    if annotation_df is None or annotation_df.empty:
        df['behavior'] = None
        return df

    ann = annotation_df.copy()
    
    # Lógica para converter formato de Intervalo (start/stop) para Frame-a-Frame
    if all(c in ann.columns for c in ['start_frame', 'stop_frame', 'action']):
        expanded = []
        for _, row in ann.iterrows():
            try:
                for f in range(int(row['start_frame']), int(row['stop_frame']) + 1):
                    expanded.append({'frame': f, 'behavior': row['action']})
            except: continue
        ann = pd.DataFrame(expanded) if expanded else pd.DataFrame(columns=['frame', 'behavior'])

    # Padronização de colunas
    if 'frame' not in ann.columns and 'frame_id' in ann.columns:
        ann = ann.rename(columns={'frame_id': 'frame'})
    
    # Identifica coluna de comportamento
    behavior_col = next((c for c in ann.columns if c in ['behavior', 'action', 'annotation']), None)
    
    if not behavior_col or ann.empty:
        df['behavior'] = None
        return df

    ann = ann[['frame', behavior_col]].rename(columns={behavior_col: 'behavior'})
    
    # Agrupa comportamentos duplicados no mesmo frame
    ann_grouped = ann.groupby('frame')['behavior'].apply(_aggregate_behaviors).reset_index()

    # Merge
    df = df.merge(ann_grouped, on='frame', how='left')
    return df

def inject_metadata_columns(df: pd.DataFrame, sequence_id: str, lab_name: str, metadata_lookup: Dict) -> pd.DataFrame:
    try: seq_int = int(sequence_id)
    except: seq_int = None
    
    meta = metadata_lookup.get(seq_int, {})
    df['video_id'] = f"{lab_name}_{sequence_id}"
    
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
            
    # Garante colunas essenciais para cálculo
    if 'pix_per_cm_approx' not in df.columns: df['pix_per_cm_approx'] = 1.0
    if 'frames_per_second' not in df.columns: df['frames_per_second'] = 30.0
    
    return df

# =================================================================
# 4. PIPELINE DE FEATURE ENGINEERING (ROBUSTO)
# =================================================================

def calculate_angle(x1, y1, x2, y2):
    """Calcula ângulo em radianos entre dois pontos."""
    return np.arctan2(y2 - y1, x2 - x1)

def pipeline_feature_engineering_robust(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo que gera features cinemáticas, sociais e temporais.
    """
    if df.empty: return pd.DataFrame()
    
    # Trabalha em uma cópia para evitar FragmentedFrameError
    df = df.copy()
    
    # 1. Tratamento Inicial de Coordenadas
    # ---------------------------------------------------------
    coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Interpolação para corrigir falhas de tracking
        df[col] = df[col].interpolate(method='linear', limit=10, limit_direction='both')

    # Dataframe separado para features novas (evita fragmentação de memória)
    feats = pd.DataFrame(index=df.index)
    
    pix_per_cm = df['pix_per_cm_approx'].replace(0, np.nan).fillna(1.0)
    
    # 2. Features Individuais (Velocidade, Aceleração, Ângulos)
    # ---------------------------------------------------------
    for m in range(1, 5): # Para cada rato (1 a 4)
        mouse_prefix = f"mouse{m}"
        
        # Tenta encontrar centro do corpo (prioriza 'body_center', 'spine', 'centroid')
        cols_x = [c for c in df.columns if c.startswith(mouse_prefix) and c.endswith('_x')]
        cols_y = [c for c in df.columns if c.startswith(mouse_prefix) and c.endswith('_y')]
        
        if not cols_x: continue

        # Define centróide (usa body_center se tiver, senão o primeiro keypoint achado)
        centroid_x_col = next((c for c in cols_x if 'body_center' in c), cols_x[0])
        centroid_y_col = next((c for c in cols_y if 'body_center' in c), cols_y[0])
        
        # Converte para CM
        cx = df[centroid_x_col] / pix_per_cm
        cy = df[centroid_y_col] / pix_per_cm
        
        feats[f'{mouse_prefix}_x_cm'] = cx
        feats[f'{mouse_prefix}_y_cm'] = cy
        
        # Velocidade
        vx = cx.diff().fillna(0)
        vy = cy.diff().fillna(0)
        speed = np.sqrt(vx**2 + vy**2)
        feats[f'{mouse_prefix}_speed_cm'] = speed
        
        # Aceleração (variação da velocidade)
        feats[f'{mouse_prefix}_accel'] = speed.diff().fillna(0)
        
        # Ângulo de Orientação (Cabeça vs Corpo)
        # Tenta achar nariz e base da cauda para direção precisa
        nose_x = next((c for c in cols_x if 'nose' in c), None)
        tail_x = next((c for c in cols_x if 'tail' in c or 'spine' in c), None)
        
        if nose_x and tail_x:
            nose_y = nose_x.replace('_x', '_y')
            tail_y = tail_x.replace('_x', '_y')
            
            # Calcula ângulo (em radianos)
            feats[f'{mouse_prefix}_angle'] = calculate_angle(
                df[tail_x], df[tail_y], df[nose_x], df[nose_y]
            ).fillna(0)
            
            # Velocidade Angular (o quanto ele gira)
            feats[f'{mouse_prefix}_angular_vel'] = feats[f'{mouse_prefix}_angle'].diff().fillna(0)

    # 3. Features Sociais (Interação entre Ratos)
    # ---------------------------------------------------------
    active_mice = [m for m in range(1, 5) if f'mouse{m}_x_cm' in feats.columns]
    
    import itertools
    for m1, m2 in itertools.combinations(active_mice, 2):
        prefix = f"m{m1}_m{m2}"
        
        x1, y1 = feats[f'mouse{m1}_x_cm'], feats[f'mouse{m1}_y_cm']
        x2, y2 = feats[f'mouse{m2}_x_cm'], feats[f'mouse{m2}_y_cm']
        
        # Distância Euclidiana
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        feats[f'dist_{prefix}'] = dist
        
        # Velocidade Relativa (estão se aproximando ou afastando?)
        # Derivada da distância: negativo = aproximando, positivo = afastando
        feats[f'rel_vel_{prefix}'] = dist.diff().fillna(0)
        
        # Ratos estão de frente um pro outro? (Diferença de ângulos)
        if f'mouse{m1}_angle' in feats and f'mouse{m2}_angle' in feats:
            angle_diff = np.abs(feats[f'mouse{m1}_angle'] - feats[f'mouse{m2}_angle'])
            # Normaliza para [0, pi]
            angle_diff = np.abs((angle_diff + np.pi) % (2 * np.pi) - np.pi)
            feats[f'angle_diff_{prefix}'] = angle_diff

    # 4. Features Temporais (Lags e Rolling Windows)
    # ---------------------------------------------------------
    # Aplica apenas nas colunas de cinemática e distância para não explodir a memória
    target_cols_for_temporal = [c for c in feats.columns if 'speed' in c or 'dist' in c or 'angle' in c]
    
    # Limita a 10 colunas mais importantes se tiver muitas, para economizar tempo/memória
    if len(target_cols_for_temporal) > 20:
        target_cols_for_temporal = [c for c in target_cols_for_temporal if 'speed' in c or 'dist' in c][:20]

    for col in target_cols_for_temporal:
        # Lags (Valores passados)
        for lag in LAG_FEATURES:
            feats[f'{col}_lag{lag}'] = feats[col].shift(lag).fillna(0)
        
        # Rolling Stats (Tendência recente)
        for window in TEMPORAL_WINDOWS:
            rolling = feats[col].rolling(window=window, min_periods=1)
            feats[f'{col}_mean_{window}'] = rolling.mean().fillna(0)
            feats[f'{col}_std_{window}'] = rolling.std().fillna(0)
            # Max pode ser útil para detectar picos de agressividade (ataques rápidos)
            if 'speed' in col:
                feats[f'{col}_max_{window}'] = rolling.max().fillna(0)

    # 5. Consolidação Final
    # ---------------------------------------------------------
    # Garante que metadados e target estão presentes
    cols_meta_present = [c for c in METADATA_COLUMNS_TO_KEEP_RAW + ['behavior'] if c in df.columns]
    
    # Concatena features novas com metadados originais
    df_final = pd.concat([df[cols_meta_present], feats], axis=1)
    
    # Tratamento final de memória
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    # Converte tipos para economizar espaço
    fcols = df_final.select_dtypes('float').columns
    df_final[fcols] = df_final[fcols].astype('float32')
    
    return df_final

# =================================================================
# 5. EXECUÇÃO PRINCIPAL
# =================================================================

if __name__ == "__main__":
    # Caminhos
    TRACKING_ROOT = Path("MABe-mouse-behavior-detection/train_tracking")
    ANNOTATIONS_ROOT = Path("MABe-mouse-behavior-detection/train_annotation")
    SEQUENCE_METADATA_PATH = Path("MABe-mouse-behavior-detection/sequence_metadata.csv")
    OUTPUT_PATH = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost")

    # Setup
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    metadata_lookup = build_metadata_lookup(load_sequence_metadata(SEQUENCE_METADATA_PATH))
    tracking_files = list(TRACKING_ROOT.rglob("*.parquet"))

    if not tracking_files:
        print(f"❌ Erro: Nenhum arquivo em {TRACKING_ROOT}")
        exit()

    print(f"🚀 Iniciando Processamento ROBUSTO de {len(tracking_files)} arquivos...")
    print(f"ℹ️  Isso vai demorar mais que o anterior, pois calcula estatísticas complexas.")

    success, failed = 0, 0
    
    # Loop de processamento
    for file_path in tqdm(tracking_files, desc="Gerando Features"):
        try:
            # 1. Leitura e Preparação
            df_raw = pd.read_parquet(file_path, engine='fastparquet')
            
            # CRÍTICO: Converte Long -> Wide imediatamente
            df_raw = convert_long_to_wide(df_raw)
            
            # Metadata e Merge de Anotações
            lab_name = file_path.parent.name if file_path.parent.name != 'train_tracking' else 'Unknown'
            seq_id = file_path.stem
            
            ann_df = load_annotation_file(seq_id, lab_name, ANNOTATIONS_ROOT)
            df_merged = merge_tracking_and_annotations(df_raw, ann_df)
            df_enriched = inject_metadata_columns(df_merged, seq_id, lab_name, metadata_lookup)
            
            # 2. Pipeline de Features Pesado
            df_final = pipeline_feature_engineering_robust(df_enriched)
            
            if df_final.empty:
                failed += 1
                continue

            # 3. Salvamento
            rel_path = file_path.relative_to(TRACKING_ROOT)
            out_file = OUTPUT_PATH / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Salva em Parquet com compressão para não ocupar disco demais
            df_final.to_parquet(out_file, index=False, compression='snappy')
            success += 1

        except Exception as e:
            # tqdm.write(f"⚠️ Falha em {file_path.name}: {e}") # Descomente para debug
            failed += 1

    print("\n" + "="*50)
    print(f"✅ FINALIZADO. Sucesso: {success} | Falhas: {failed}")
    print("="*50)
    print("Agora rode: python consolidate_data_catboost.py")