import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# CONFIGURAÇÕES PARA CATBOOST
# =================================================================

# Colunas de metadados que queremos manter
METADATA_COLUMNS_TO_KEEP_RAW = [
    'frame', 'behavior', 'video_id', 'unique_frame_id',
    'lab_id', 
    'frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 
    'video_width_pix', 'video_height_pix', 
    'body_parts_tracked', 'behaviors_labeled', 'tracking_method',
    'mouse1_strain', 'mouse1_color', 'mouse1_condition', 'mouse1_id',
    'mouse2_strain', 'mouse2_color', 'mouse2_condition', 'mouse2_id',
    'mouse3_strain', 'mouse3_color', 'mouse3_condition', 'mouse3_id',
    'mouse4_strain', 'mouse4_color', 'mouse4_condition', 'mouse4_id',
]

# Variáveis categóricas (mantidas como categóricas para CatBoost)
CATEGORICAL_COLS = [
    'arena_type', 'arena_shape',
    'mouse1_sex', 'mouse2_sex', 'mouse3_sex', 'mouse4_sex',
    'mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain',
    'mouse1_color', 'mouse2_color', 'mouse3_color', 'mouse4_color',
    'mouse1_condition', 'mouse2_condition', 'mouse3_condition', 'mouse4_condition',
    'lab_id', 'tracking_method',
]

# Variáveis numéricas de contexto
NUMERIC_CONTEXT_COLS = [
    'mouse1_age', 'mouse2_age', 'mouse3_age', 'mouse4_age',
    'arena_width_cm', 'arena_height_cm',
    'frames_per_second', 'video_duration_sec', 'pix_per_cm_approx',
    'video_width_pix', 'video_height_pix',
]

# Parâmetros para features temporais
TEMPORAL_WINDOWS = [3, 5, 10, 20]  # Janelas para rolling statistics
LAG_FEATURES = [1, 2, 3, 5, 10]  # Lags temporais


# =================================================================
# FUNÇÕES AUXILIARES
# =================================================================

def safe_interpolate(series: pd.Series, limit: int = 10) -> pd.Series:
    """Interpolação linear segura com limite."""
    return series.interpolate(method='linear', limit=limit, limit_direction='both')


def load_sequence_metadata(metadata_path: Path) -> pd.DataFrame:
    """Carrega o arquivo sequence_metadata.csv se existir."""
    if not metadata_path or not metadata_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(metadata_path)
    except Exception as exc:
        print(f"⚠️ Não foi possível carregar {metadata_path}: {exc}")
        return pd.DataFrame()


def build_metadata_lookup(metadata_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Cria um mapa sequence_id -> metadados."""
    if metadata_df.empty or 'sequence_id' not in metadata_df.columns:
        return {}

    metadata_df = metadata_df.set_index('sequence_id')
    lookup: Dict[int, Dict[str, Any]] = {}
    for seq_id, row in metadata_df.iterrows():
        try:
            lookup[int(seq_id)] = row.to_dict()
        except (ValueError, TypeError):
            continue
    return lookup


def load_annotation_file(
    sequence_id: str,
    lab_name: str,
    annotations_root: Optional[Path],
    master_annotations_path: Optional[Path],
) -> pd.DataFrame:
    """Carrega o arquivo de anotações correspondente ao sequence_id."""
    candidate_files: List[Path] = []

    if annotations_root and annotations_root.exists():
        candidate_files.append(annotations_root / lab_name / f"{sequence_id}.parquet")
        candidate_files.append(annotations_root / f"{sequence_id}.parquet")

    for candidate in candidate_files:
        if candidate.exists():
            try:
                return pd.read_parquet(candidate, engine='fastparquet')
            except Exception as exc:
                print(f"⚠️ Falha ao ler {candidate}: {exc}")

    if master_annotations_path and master_annotations_path.exists():
        try:
            seq_int = int(sequence_id)
        except ValueError:
            seq_int = sequence_id
        try:
            df_master = pd.read_parquet(
                master_annotations_path,
                engine='pyarrow',
                filters=[('sequence_id', '==', seq_int)],
            )
            if 'sequence_id' in df_master.columns:
                df_master = df_master.drop(columns=['sequence_id'])
            return df_master
        except Exception as exc:
            print(f"⚠️ Falha ao filtrar {master_annotations_path.name} para {sequence_id}: {exc}")

    return pd.DataFrame()


def _aggregate_behaviors(values: pd.Series) -> Any:
    """Agrupa múltiplos comportamentos no mesmo frame."""
    behaviors: List[str] = []
    for value in values:
        if isinstance(value, list):
            behaviors.extend([str(v).strip() for v in value if str(v).strip()])
        elif isinstance(value, (tuple, set)):
            behaviors.extend([str(v).strip() for v in value if str(v).strip()])
        elif pd.isna(value):
            continue
        else:
            str_val = str(value).strip()
            if not str_val:
                continue
            if ';' in str_val:
                behaviors.extend([v.strip() for v in str_val.split(';') if v.strip()])
            else:
                behaviors.append(str_val)

    if not behaviors:
        return np.nan

    unique_vals = sorted(set(behaviors))
    if len(unique_vals) == 1:
        return unique_vals[0]
    return unique_vals


def merge_tracking_and_annotations(
    tracking_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combina rastreamento com anotações em um único DataFrame."""
    df = tracking_df.copy()

    if 'frame' not in df.columns:
        df['frame'] = np.arange(len(df), dtype=np.int32)

    if annotation_df is None or annotation_df.empty:
        if 'behavior' not in df.columns:
            df['behavior'] = None
        return df

    ann = annotation_df.copy()
    if 'frame' not in ann.columns:
        ann['frame'] = np.arange(len(ann), dtype=np.int32)

    behavior_col = 'behavior'
    if behavior_col not in ann.columns:
        candidates = [col for col in ann.columns if 'behavior' in col.lower()]
        if candidates:
            behavior_col = candidates[0]
        else:
            if 'behavior' not in df.columns:
                df['behavior'] = None
            return df

    ann = ann[['frame', behavior_col]].rename(columns={behavior_col: 'behavior'})
    ann_grouped = (
        ann.groupby('frame')['behavior']
        .apply(_aggregate_behaviors)
        .reset_index()
    )

    df = df.merge(ann_grouped, on='frame', how='left', suffixes=('', '_ann'))

    if 'behavior_ann' in df.columns:
        if 'behavior' in df.columns:
            df['behavior'] = df['behavior_ann'].combine_first(df['behavior'])
        else:
            df['behavior'] = df['behavior_ann']
        df.drop(columns=['behavior_ann'], inplace=True)

    if 'behavior' not in df.columns:
        df['behavior'] = None

    return df


def inject_metadata_columns(
    df: pd.DataFrame,
    sequence_id: str,
    lab_name: str,
    metadata_lookup: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """Adiciona colunas de metadados usando sequence_metadata.csv."""
    df = df.copy()

    try:
        seq_int = int(sequence_id)
    except (ValueError, TypeError):
        seq_int = None

    metadata_values = metadata_lookup.get(seq_int, {}) if seq_int is not None else {}

    df['lab_id'] = df.get('lab_id', lab_name)
    df['video_id'] = df.get('video_id', f"{lab_name}_{sequence_id}")

    for col, value in metadata_values.items():
        if col in df.columns:
            continue
        df[col] = value

    defaults = {
        'pix_per_cm_approx': 1.0,
        'frames_per_second': 30.0,
        'video_width_pix': np.nan,
        'video_height_pix': np.nan,
    }
    for col, default_value in defaults.items():
        if col not in df.columns:
            df[col] = default_value

    if 'video_duration_sec' not in df.columns:
        fps = pd.to_numeric(df['frames_per_second'], errors='coerce').fillna(30.0)
        if not df.empty:
            df['video_duration_sec'] = df['frame'].max() / max(fps.iloc[0], 1.0)
        else:
            df['video_duration_sec'] = np.nan

    if 'unique_frame_id' not in df.columns:
        df['unique_frame_id'] = df['video_id'].astype(str) + "_" + df['frame'].astype(str)

    return df


def prepare_tracking_dataframe(
    tracking_path: Path,
    annotations_root: Optional[Path],
    master_annotations_path: Optional[Path],
    metadata_lookup: Dict[int, Dict[str, Any]],
    tracking_root: Path,
) -> pd.DataFrame:
    """Carrega o arquivo de tracking e injeta anotações + metadados."""
    try:
        df_tracking = pd.read_parquet(tracking_path, engine='fastparquet')
    except Exception as exc:
        print(f"⚠️ Falha ao ler {tracking_path.name}: {exc}")
        return pd.DataFrame()

    try:
        relative_parts = tracking_path.relative_to(tracking_root).parts
        lab_name = relative_parts[0]
    except ValueError:
        lab_name = tracking_path.parent.name

    sequence_id = tracking_path.stem
    annotation_df = load_annotation_file(
        sequence_id=sequence_id,
        lab_name=lab_name,
        annotations_root=annotations_root,
        master_annotations_path=master_annotations_path,
    )

    df_combined = merge_tracking_and_annotations(df_tracking, annotation_df)
    df_enriched = inject_metadata_columns(
        df=df_combined,
        sequence_id=sequence_id,
        lab_name=lab_name,
        metadata_lookup=metadata_lookup,
    )
    return df_enriched




def add_temporal_features(df: pd.DataFrame, group_col: str = 'video_id') -> pd.DataFrame:
    """
    Adiciona features temporais essenciais para modelos de árvore:
    - Lags (valores anteriores)
    - Rolling statistics (média, std, min, max)
    - Diferenças temporais
    """
    df = df.copy()
    df = df.sort_values([group_col, 'frame']).reset_index(drop=True)
    
    # Features de movimento e distância para aplicar lags
    movement_cols = [col for col in df.columns if 'speed' in col or 'dist_' in col or 'accel' in col]
    coord_cols = [col for col in df.columns if col.endswith('_cm_x') or col.endswith('_cm_y')]
    
    all_temporal_cols = movement_cols + coord_cols[:20]  # Limita para não criar muitas features
    
    new_features = []
    
    for col in all_temporal_cols:
        if col not in df.columns:
            continue
            
        # Lags temporais
        for lag in LAG_FEATURES:
            lag_col = f'{col}_lag{lag}'
            df[lag_col] = df.groupby(group_col)[col].shift(lag)
            new_features.append(lag_col)
        
        # Rolling statistics
        for window in TEMPORAL_WINDOWS:
            grouped = df.groupby(group_col)[col]
            
            # Média móvel
            mean_col = f'{col}_rolling_mean_{window}'
            df[mean_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
            new_features.append(mean_col)
            
            # Desvio padrão móvel
            std_col = f'{col}_rolling_std_{window}'
            df[std_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
            new_features.append(std_col)
            
            # Máximo móvel
            max_col = f'{col}_rolling_max_{window}'
            df[max_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
            new_features.append(max_col)
            
            # Mínimo móvel
            min_col = f'{col}_rolling_min_{window}'
            df[min_col] = grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
            new_features.append(min_col)
        
        # Diferença temporal (derivada)
        diff_col = f'{col}_diff'
        df[diff_col] = df.groupby(group_col)[col].diff()
        new_features.append(diff_col)
    
    return df, new_features


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona features de interação entre ratos (útil para comportamento social).
    """
    df = df.copy()
    new_features = []
    
    # Distâncias entre ratos (já calculadas, mas podemos adicionar mais)
    mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    
    # Velocidades relativas
    for m1, m2 in mouse_pairs:
        speed1_col = f'mouse{m1}_speed_cm_per_frame'
        speed2_col = f'mouse{m2}_speed_cm_per_frame'
        dist_col = f'dist_m{m1}_m{m2}_cm'
        
        if all(col in df.columns for col in [speed1_col, speed2_col]):
            rel_speed_col = f'rel_speed_m{m1}_m{m2}'
            df[rel_speed_col] = np.abs(df[speed1_col] - df[speed2_col])
            new_features.append(rel_speed_col)
        
        # Razão de velocidade sobre distância (proximidade relativa)
        if all(col in df.columns for col in [speed1_col, speed2_col, dist_col]):
            proximity_col = f'proximity_speed_m{m1}_m{m2}'
            df[proximity_col] = (df[speed1_col] + df[speed2_col]) / (df[dist_col] + 1e-6)
            new_features.append(proximity_col)
    
    return df, new_features


# =================================================================
# PIPELINE PRINCIPAL PARA CATBOOST
# =================================================================

def pipeline_feature_engineering_catboost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de feature engineering otimizado para CatBoost.
    
    Diferenças principais em relação ao pipeline original:
    1. Mantém variáveis categóricas como categóricas (sem OHE)
    2. Não normaliza features numéricas (CatBoost não precisa)
    3. Adiciona features temporais (lags, rolling stats)
    4. Trata valores ausentes de forma adequada para CatBoost
    """
    
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    
    # --------------------------------------------------------------------------------
    # 1. Pré-processamento e Normalização CM (mantido do pipeline original)
    # --------------------------------------------------------------------------------
    coord_cols_raw = [col for col in df.columns 
                      if (col.endswith('_x') or col.endswith('_y')) and col.startswith('mouse')]

    # Garante que as colunas críticas são numéricas
    for col in coord_cols_raw:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    if 'pix_per_cm_approx' in df.columns:
        df.loc[:, 'pix_per_cm_approx'] = pd.to_numeric(df['pix_per_cm_approx'], errors='coerce').astype(float)
    
    # Padroniza 0.0 como NaN
    df.loc[:, coord_cols_raw] = df.loc[:, coord_cols_raw].replace(0.0, np.nan)
    
    # Interpolação Linear
    for col in coord_cols_raw:
        df.loc[:, col] = df.groupby('video_id')[col].transform(safe_interpolate)
    
    # Calcula o centro do corpo
    center_cols_pix = []
    for m in range(1, 5):
        hip_left_x = f'mouse{m}_hip_left_x'
        hip_right_x = f'mouse{m}_hip_right_x'
        hip_left_y = f'mouse{m}_hip_left_y'
        hip_right_y = f'mouse{m}_hip_right_y'
        
        center_x = f'mouse{m}_body_center_x'
        center_y = f'mouse{m}_body_center_y'

        if hip_left_x in df.columns and hip_right_x in df.columns:
            df.loc[:, center_x] = ((df[hip_left_x] + df[hip_right_x]) / 2.0).astype(float)
            df.loc[:, center_y] = ((df[hip_left_y] + df[hip_right_y]) / 2.0).astype(float)
            center_cols_pix.extend([center_x, center_y])
    
    all_coord_cols_pix = coord_cols_raw + center_cols_pix

    # Normalização: Pixels para Centímetros (CM)
    df_cm = pd.DataFrame(index=df.index)
    cm_cols = []
    for col in all_coord_cols_pix:
        col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
        df_cm.loc[:, col_cm] = (df[col] / df['pix_per_cm_approx']).astype(float)
        cm_cols.append(col_cm)

    # --------------------------------------------------------------------------------
    # 2. Geração de Features de Velocidade, Distância e Aceleração
    # --------------------------------------------------------------------------------
    df_kinematics = pd.DataFrame(index=df.index)
    speed_cols = []
    
    # Velocidade
    for m in range(1, 5):
        center_x_cm = f'mouse{m}_body_center_cm_x'
        center_y_cm = f'mouse{m}_body_center_cm_y'
        
        if center_x_cm in df_cm.columns:
            delta_x = df_cm.groupby(df['video_id'])[center_x_cm].diff()
            delta_y = df_cm.groupby(df['video_id'])[center_y_cm].diff()
            
            speed_col_name = f'mouse{m}_speed_cm_per_frame'
            df_kinematics.loc[:, speed_col_name] = np.sqrt(delta_x**2 + delta_y**2)
            speed_cols.append(speed_col_name)

    # Distância Social
    mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    dist_cols = []
    for m1, m2 in mouse_pairs:
        dist_col_name = f'dist_m{m1}_m{m2}_cm'
        center1_x = f'mouse{m1}_body_center_cm_x'
        center1_y = f'mouse{m1}_body_center_cm_y'
        center2_x = f'mouse{m2}_body_center_cm_x'
        center2_y = f'mouse{m2}_body_center_cm_y'
        
        if center1_x in df_cm.columns and center2_x in df_cm.columns:
            df_kinematics.loc[:, dist_col_name] = np.sqrt(
                (df_cm[center1_x] - df_cm[center2_x])**2 +
                (df_cm[center1_y] - df_cm[center2_y])**2
            )
            dist_cols.append(dist_col_name)
    
    # Features de Postura e Ângulo
    angle_cols = []
    for m in range(1, 5):
        tail_x = f'mouse{m}_tail_base_cm_x'
        tail_y = f'mouse{m}_tail_base_cm_y'
        nose_x = f'mouse{m}_nose_cm_x'
        nose_y = f'mouse{m}_nose_cm_y'
        
        if tail_x in df_cm.columns and nose_x in df_cm.columns:
            dx = df_cm[nose_x] - df_cm[tail_x]
            dy = df_cm[nose_y] - df_cm[tail_y]
            
            angle_col_name = f'mouse{m}_body_angle_rad'
            df_kinematics.loc[:, angle_col_name] = np.arctan2(dy, dx)
            angle_cols.append(angle_col_name)

    # Features de Aceleração
    accel_cols = []
    for speed_col in speed_cols:
        accel_col_name = speed_col.replace('speed', 'accel')
        df_kinematics.loc[:, accel_col_name] = df_kinematics.groupby(df['video_id'])[speed_col].diff()
        accel_cols.append(accel_col_name)

    # Features de Distância a Pontos Chave
    point_dist_cols = []
    for m1, m2 in mouse_pairs:
        nose1_x, nose1_y = f'mouse{m1}_nose_cm_x', f'mouse{m1}_nose_cm_y'
        tail2_x, tail2_y = f'mouse{m2}_tail_base_cm_x', f'mouse{m2}_tail_base_cm_y'
        nose2_x, nose2_y = f'mouse{m2}_nose_cm_x', f'mouse{m2}_nose_cm_y'
        tail1_x, tail1_y = f'mouse{m1}_tail_base_cm_x', f'mouse{m1}_tail_base_cm_y'
        
        if nose1_x in df_cm.columns and tail2_x in df_cm.columns:
            dist_col_name_1_2 = f'dist_nose{m1}_tail{m2}_cm'
            df_kinematics.loc[:, dist_col_name_1_2] = np.sqrt(
                (df_cm[nose1_x] - df_cm[tail2_x])**2 +
                (df_cm[nose1_y] - df_cm[tail2_y])**2
            )
            point_dist_cols.append(dist_col_name_1_2)

        if nose2_x in df_cm.columns and tail1_x in df_cm.columns:
            dist_col_name_2_1 = f'dist_nose{m2}_tail{m1}_cm'
            df_kinematics.loc[:, dist_col_name_2_1] = np.sqrt(
                (df_cm[nose2_x] - df_cm[tail1_x])**2 +
                (df_cm[nose2_y] - df_cm[tail1_y])**2
            )
            point_dist_cols.append(dist_col_name_2_1)

    # --------------------------------------------------------------------------------
    # 3. Features Temporais (NOVO - Essencial para CatBoost com dados temporais)
    # --------------------------------------------------------------------------------
    # Concatena features básicas antes de adicionar temporais
    df_temp = pd.concat([df_cm, df_kinematics], axis=1)
    df_temp['video_id'] = df['video_id'].values
    df_temp['frame'] = df['frame'].values if 'frame' in df.columns else df.index
    
    # Adiciona features temporais
    df_temp, temporal_features = add_temporal_features(df_temp, group_col='video_id')
    
    # Adiciona features de interação
    df_temp, interaction_features = add_interaction_features(df_temp)
    
    # --------------------------------------------------------------------------------
    # 4. Processamento de Metadados (OTIMIZADO PARA CATBOOST)
    # --------------------------------------------------------------------------------
    # Mantém variáveis categóricas como categóricas (sem OHE)
    categorical_cols_present = [col for col in CATEGORICAL_COLS if col in df.columns]
    
    # Converte para tipo categórico (CatBoost funciona melhor assim)
    df_context = df[categorical_cols_present + NUMERIC_CONTEXT_COLS].copy()
    
    for col in categorical_cols_present:
        df_context[col] = df_context[col].astype('category')
    
    # Variáveis numéricas de contexto (sem normalização - CatBoost não precisa)
    for col in NUMERIC_CONTEXT_COLS:
        if col in df_context.columns:
            df_context.loc[:, col] = pd.to_numeric(df_context[col], errors='coerce').astype(float)
    
    # --------------------------------------------------------------------------------
    # 5. Concatenação Final
    # --------------------------------------------------------------------------------
    base_cols = list(set(METADATA_COLUMNS_TO_KEEP_RAW) & set(df.columns))
    df_final = df[base_cols].copy()

    # Seleciona apenas as colunas numéricas de df_temp (exclui video_id e frame que já estão em base_cols)
    temp_numeric_cols = [col for col in df_temp.columns 
                        if col not in ['video_id', 'frame'] and df_temp[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
    
    df_final = pd.concat([
        df_final,
        df_temp[temp_numeric_cols],
        df_context
    ], axis=1)
    
    return df_final.reset_index(drop=True)


# =================================================================
# EXECUÇÃO DA PIPELINE
# =================================================================

if __name__ == "__main__":
    TRACKING_ROOT = Path("MABe-mouse-behavior-detection/train_tracking")
    ANNOTATIONS_ROOT = Path("MABe-mouse-behavior-detection/train_annotation")
    MASTER_ANNOTATIONS_PATH = Path("MABe-mouse-behavior-detection/train_annotations.parquet")
    SEQUENCE_METADATA_PATH = Path("MABe-mouse-behavior-detection/sequence_metadata.csv")
    OUTPUT_PATH = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost")

    metadata_df = load_sequence_metadata(SEQUENCE_METADATA_PATH)
    metadata_lookup = build_metadata_lookup(metadata_df)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    if not TRACKING_ROOT.exists():
        raise FileNotFoundError(
            f"Pasta de tracking não encontrada: {TRACKING_ROOT}. "
            "Verifique se o dataset do Kaggle está no caminho correto."
        )

    tracking_files = list(TRACKING_ROOT.rglob("*.parquet"))

    if not tracking_files:
        print(f"❌ Nenhum arquivo de tracking encontrado em {TRACKING_ROOT.resolve()}")
    else:
        print(f"🔍 Encontrados {len(tracking_files)} arquivos de tracking.")
        print("🚀 Preparando dados diretamente de train_tracking/train_annotation ...")

        for file_path in tqdm(tracking_files, desc="Processando Features para CatBoost"):
            try:
                df_raw = prepare_tracking_dataframe(
                    tracking_path=file_path,
                    annotations_root=ANNOTATIONS_ROOT,
                    master_annotations_path=MASTER_ANNOTATIONS_PATH,
                    metadata_lookup=metadata_lookup,
                    tracking_root=TRACKING_ROOT,
                )

                if df_raw.empty:
                    tqdm.write(f"⚠️ Dados vazios em {file_path.name}. Pulando.")
                    continue

                df_processed = pipeline_feature_engineering_catboost(df_raw)

                relative_path = file_path.relative_to(TRACKING_ROOT)
                output_file = OUTPUT_PATH / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                df_processed.to_parquet(output_file, engine='fastparquet', index=False)

            except Exception as e:
                print(f"\n⚠️ ERRO ao processar {file_path}: {e}. Pulando.")
                import traceback
                traceback.print_exc()
                continue

        print("\n✅ Processamento de Features para CatBoost concluído.")
        print(f"📁 Dados processados salvos em: {OUTPUT_PATH.resolve()}")
        print("\n💡 Próximos passos:")
        print("   1. Execute consolidate_data_catboost.py para consolidar os dados")
        print("   2. Use as variáveis categóricas diretamente no CatBoost (sem OHE)")
        print("   3. CatBoost lida automaticamente com valores ausentes")


