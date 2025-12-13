import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
import warnings
import gc
import re

warnings.filterwarnings('ignore')

# =================================================================
# 1. CONFIGURAÇÕES E CONSTANTES
# =================================================================

# Lista completa baseada no seu train.csv
METADATA_COLUMNS_TO_KEEP_RAW = [
    'frame', 'behavior', 'video_id', 'unique_frame_id',
    'lab_id', 'frames_per_second', 'video_duration_sec', 
    'pix_per_cm_approx', 'video_width_pix', 'video_height_pix', 
    'num_mice', 'arena_type', 'arena_shape', 'tracking_method',
    'arena_width_cm', 'arena_height_cm',
    # Mouse 1
    'mouse1_strain', 'mouse1_sex', 'mouse1_age', 'mouse1_id', 'mouse1_color', 'mouse1_condition',
    # Mouse 2
    'mouse2_strain', 'mouse2_sex', 'mouse2_age', 'mouse2_id', 'mouse2_color', 'mouse2_condition',
    # Mouse 3
    'mouse3_strain', 'mouse3_sex', 'mouse3_age', 'mouse3_id', 'mouse3_color', 'mouse3_condition',
    # Mouse 4
    'mouse4_strain', 'mouse4_sex', 'mouse4_age', 'mouse4_id', 'mouse4_color', 'mouse4_condition'
]

# Janelas temporais
TEMPORAL_WINDOWS = [5, 15, 30]

# Lags
LAG_FEATURES = [1, 5, 10]

# =================================================================
# 2. FUNÇÕES DE CORREÇÃO DE FORMATO
# =================================================================

def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['bodypart', 'x', 'y', 'mouse_id']
    if not all(col in df.columns for col in required_cols):
        return df

    df['mouse_key'] = 'mouse' + df['mouse_id'].astype(str)
    
    try:
        pivot_df = df.pivot_table(
            index='video_frame', 
            columns=['mouse_key', 'bodypart'], 
            values=['x', 'y']
        )
        new_columns = [f"{m}_{p}_{v}" for v, m, p in pivot_df.columns]
        pivot_df.columns = new_columns
        pivot_df = pivot_df.reset_index()
        pivot_df = pivot_df.rename(columns={'video_frame': 'frame'})
        return pivot_df
    except:
        return df

# =================================================================
# 3. FUNÇÕES AUXILIARES DE CARREGAMENTO
# =================================================================

def load_sequence_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path or not metadata_path.exists():
        print(f"⚠️ Aviso: {metadata_path} não encontrado.")
        return pd.DataFrame()
    try:
        return pd.read_csv(metadata_path)
    except Exception:
        return pd.DataFrame()

def build_metadata_lookup(metadata_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if metadata_df.empty: return {}
    
    # Se tiver video_id, usa ele (padrao do train.csv)
    if 'video_id' in metadata_df.columns:
        metadata_df['video_id'] = metadata_df['video_id'].astype(str)
        return metadata_df.set_index('video_id').to_dict('index')
    
    # Fallback
    if 'sequence_id' in metadata_df.columns:
        metadata_df['sequence_id'] = metadata_df['sequence_id'].astype(str)
        return metadata_df.set_index('sequence_id').to_dict('index')
        
    return {}

def load_annotation_file(sequence_id: str, lab_name: str, annotations_root: Optional[Path]) -> pd.DataFrame:
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
    behaviors = set()
    for value in values:
        val_str = str(value).strip()
        if not val_str or val_str.lower() in ['nan', 'none']:
            continue
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
    
    if all(c in ann.columns for c in ['start_frame', 'stop_frame', 'action']):
        expanded = []
        for _, row in ann.iterrows():
            try:
                for f in range(int(row['start_frame']), int(row['stop_frame']) + 1):
                    expanded.append({'frame': f, 'behavior': row['action']})
            except: continue
        ann = pd.DataFrame(expanded) if expanded else pd.DataFrame(columns=['frame', 'behavior'])

    if 'frame' not in ann.columns and 'frame_id' in ann.columns:
        ann = ann.rename(columns={'frame_id': 'frame'})
    
    behavior_col = next((c for c in ann.columns if c in ['behavior', 'action', 'annotation']), None)
    
    if not behavior_col or ann.empty:
        df['behavior'] = None
        return df

    ann = ann[['frame', behavior_col]].rename(columns={behavior_col: 'behavior'})
    ann_grouped = ann.groupby('frame')['behavior'].apply(_aggregate_behaviors).reset_index()
    df = df.merge(ann_grouped, on='frame', how='left')
    return df

# --- CORREÇÃO AQUI: ACEITA 4 ARGUMENTOS ---
def inject_metadata_columns(df: pd.DataFrame, sequence_id: str, lab_name: str, metadata_lookup: Dict) -> pd.DataFrame:
    
    # Tenta pegar pelo ID
    meta = metadata_lookup.get(str(sequence_id), {})
    
    df['video_id'] = f"{lab_name}_{sequence_id}"
    df['lab_id'] = lab_name 
    
    # Conta ratos se não tiver no CSV
    if 'num_mice' not in meta:
        cnt = 0
        for m in range(1, 5):
            if any(c.startswith(f'mouse{m}_') for c in df.columns): cnt += 1
        meta['num_mice'] = cnt

    # Injeta dados do CSV
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
            
    # Garante colunas vazias se faltar
    for col in METADATA_COLUMNS_TO_KEEP_RAW:
        if col not in df.columns:
            if 'age' in col or 'num' in col or 'width' in col or 'height' in col or 'id' in col:
                df[col] = -1
            else:
                df[col] = 'unknown'
            
    if 'pix_per_cm_approx' not in df.columns: df['pix_per_cm_approx'] = 1.0
    if 'frames_per_second' not in df.columns: df['frames_per_second'] = 30.0
    
    return df

# =================================================================
# 4. PIPELINE DE FEATURE ENGINEERING (CORRIGIDO)
# =================================================================

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)

def pipeline_feature_engineering_robust(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    df = df.copy()
    
    coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(limit=10)

    feats = pd.DataFrame(index=df.index)
    pix_per_cm = df['pix_per_cm_approx'].replace(0, np.nan).fillna(1.0)
    
    for m in range(1, 5): 
        prefix = f"mouse{m}"
        cols_x = [c for c in df.columns if c.startswith(prefix) and c.endswith('_x')]
        cols_y = [c for c in df.columns if c.startswith(prefix) and c.endswith('_y')]
        
        if not cols_x: continue

        centroid_x_col = next((c for c in cols_x if 'body_center' in c), cols_x[0])
        centroid_y_col = next((c for c in cols_y if 'body_center' in c), cols_y[0])
        
        # --- Normalização Espacial ---
        max_w = df[centroid_x_col].max() if df[centroid_x_col].max() > 0 else 1.0
        max_h = df[centroid_y_col].max() if df[centroid_y_col].max() > 0 else 1.0
        feats[f'{prefix}_x_rel'] = df[centroid_x_col] / max_w
        feats[f'{prefix}_y_rel'] = df[centroid_y_col] / max_h
        
        cx = df[centroid_x_col] / pix_per_cm
        cy = df[centroid_y_col] / pix_per_cm
        
        feats[f'{prefix}_x_cm'] = cx
        feats[f'{prefix}_y_cm'] = cy
        
        vx = cx.diff().fillna(0)
        vy = cy.diff().fillna(0)
        speed = np.sqrt(vx**2 + vy**2)
        feats[f'{prefix}_speed_cm'] = speed
        feats[f'{prefix}_accel'] = speed.diff().fillna(0)
        
        nose_x = next((c for c in cols_x if 'nose' in c), None)
        tail_x = next((c for c in cols_x if 'tail' in c or 'spine' in c), None)
        
        if nose_x and tail_x:
            nose_y = nose_x.replace('_x', '_y')
            tail_y = tail_x.replace('_x', '_y')
            feats[f'{prefix}_angle'] = calculate_angle(df[tail_x], df[tail_y], df[nose_x], df[nose_y]).fillna(0)
            feats[f'{prefix}_angular_vel'] = feats[f'{prefix}_angle'].diff().fillna(0)

    active_mice = [m for m in range(1, 5) if f'mouse{m}_x_cm' in feats.columns]
    
    import itertools
    for m1, m2 in itertools.combinations(active_mice, 2):
        prefix = f"m{m1}_m{m2}"
        x1, y1 = feats[f'mouse{m1}_x_cm'], feats[f'mouse{m1}_y_cm']
        x2, y2 = feats[f'mouse{m2}_x_cm'], feats[f'mouse{m2}_y_cm']
        
        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        feats[f'dist_{prefix}'] = dist
        feats[f'rel_vel_{prefix}'] = dist.diff().fillna(0)
        
        if f'mouse{m1}_angle' in feats and f'mouse{m2}_angle' in feats:
            angle_diff = np.abs(feats[f'mouse{m1}_angle'] - feats[f'mouse{m2}_angle'])
            angle_diff = np.abs((angle_diff + np.pi) % (2 * np.pi) - np.pi)
            feats[f'angle_diff_{prefix}'] = angle_diff

    # --- Lags e Rolling (INDENTAÇÃO CORRIGIDA) ---
    target_cols = [c for c in feats.columns if 'speed' in c or 'dist' in c or 'angle' in c]
    if len(target_cols) > 25: target_cols = target_cols[:25]

    for col in target_cols:
        for lag in LAG_FEATURES:
            feats[f'{col}_lag{lag}'] = feats[col].shift(lag).fillna(0)
        
        for window in TEMPORAL_WINDOWS:
            rolling = feats[col].rolling(window=window, min_periods=1)
            feats[f'{col}_mean_{window}'] = rolling.mean().fillna(0)
            feats[f'{col}_std_{window}'] = rolling.std().fillna(0)
            # O 'if' agora está dentro do loop corretamente
            if 'speed' in col:
                feats[f'{col}_max_{window}'] = rolling.max().fillna(0)

    # Consolidação
    cols_meta_present = [c for c in METADATA_COLUMNS_TO_KEEP_RAW + ['behavior'] if c in df.columns]
    
    df_final = pd.concat([df[cols_meta_present], feats], axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    
    fcols = df_final.select_dtypes('float').columns
    df_final[fcols] = df_final[fcols].astype('float32')
    
    return df_final

# =================================================================
# 5. EXECUÇÃO PRINCIPAL
# =================================================================

if __name__ == "__main__":
    BASE_DIR = Path("MABe-mouse-behavior-detection")
    TRACKING_ROOT = BASE_DIR / "train_tracking"
    ANNOTATIONS_ROOT = BASE_DIR / "train_annotation"
    
    # Aponta para train.csv
    METADATA_PATH = BASE_DIR / "train.csv"
    
    OUTPUT_PATH = BASE_DIR / "feature_engineered_data_catboost_v4"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("📋 Carregando train.csv...")
    meta_df = load_sequence_metadata(METADATA_PATH)
    meta_lookup = build_metadata_lookup(meta_df)
    
    tracking_files = list(TRACKING_ROOT.rglob("*.parquet"))
    print(f"🚀 Iniciando Processamento ({len(tracking_files)} arquivos)...")

    # --- CORREÇÃO DO UNPACK 0, 0 ---
    success, failed = 0, 0
    
    for file_path in tqdm(tracking_files, desc="Gerando Features"):
        try:
            df_raw = pd.read_parquet(file_path, engine='fastparquet')
            df_raw = convert_long_to_wide(df_raw)
            
            lab_name = file_path.parent.name
            seq_id = file_path.stem
            
            ann_df = load_annotation_file(seq_id, lab_name, ANNOTATIONS_ROOT)
            df_merged = merge_tracking_and_annotations(df_raw, ann_df)
            
            # Chama com 4 argumentos
            df_enriched = inject_metadata_columns(df_merged, seq_id, lab_name, meta_lookup)
            
            df_final = pipeline_feature_engineering_robust(df_enriched)
            
            if not df_final.empty:
                out_file = OUTPUT_PATH / f"{seq_id}.parquet"
                df_final.to_parquet(out_file, index=False, compression='snappy')
                success += 1

        except Exception as e:
            # tqdm.write(f"⚠️ Falha em {file_path.name}: {e}") 
            failed += 1

    print("\n" + "="*50)
    print(f"✅ FINALIZADO. Sucesso: {success} | Falhas: {failed}")
    print("="*50)