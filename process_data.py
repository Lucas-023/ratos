import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

# Colunas de metadados BRUTOS que queremos manter no arquivo final (além das features)
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


def pipeline_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o feature engineering completo a um ÚNICO arquivo Parquet, incluindo novas features."""
    
    if df.empty:
        return pd.DataFrame()

    df = df.copy() 
    
    # --------------------------------------------------------------------------------
    # 1. Pré-processamento e Normalização CM
    # --------------------------------------------------------------------------------
    coord_cols_raw = [col for col in df.columns 
                  if (col.endswith('_x') or col.endswith('_y')) and col.startswith('mouse')]

    # 1a. Garante que as colunas críticas são numéricas 
    for col in coord_cols_raw:
        df.loc[:, col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    if 'pix_per_cm_approx' in df.columns:
        df.loc[:, 'pix_per_cm_approx'] = pd.to_numeric(df['pix_per_cm_approx'], errors='coerce').astype(float)
    
    # 1b. Padroniza 0.0 como NaN
    df.loc[:, coord_cols_raw] = df.loc[:, coord_cols_raw].replace(0.0, np.nan)

    # 1c. Interpolação Linear 
    for col in coord_cols_raw:
        df.loc[:, col] = df.groupby('video_id')[col].transform(
             lambda x: x.interpolate(method='linear', limit=10, limit_direction='both')
        )
    
    # 1d. CALCULA O CENTRO DO CORPO
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
    
    # Lista FINAL de todas as colunas de coordenadas (pixels) a serem normalizadas
    all_coord_cols_pix = coord_cols_raw + center_cols_pix

    # 1e. Normalização: Pixels para Centímetros (CM)
    df_cm = pd.DataFrame(index=df.index) 
    cm_cols = []
    for col in all_coord_cols_pix:
        col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
        # Aplica a normalização Pix -> CM
        df_cm.loc[:, col_cm] = (df[col] / df['pix_per_cm_approx']).astype(float) 
        cm_cols.append(col_cm)

    # --------------------------------------------------------------------------------
    # 2. Geração de Features de Velocidade, Distância e NOVAS FEATURES
    # --------------------------------------------------------------------------------
    
    df_kinematics = pd.DataFrame(index=df.index) 
    speed_cols = []
    
    # 2a. Velocidade (Mantida)
    for m in range(1, 5):
        center_x_cm = f'mouse{m}_body_center_cm_x'
        center_y_cm = f'mouse{m}_body_center_cm_y'
        
        if center_x_cm in df_cm.columns: 
            delta_x = df_cm.groupby(df['video_id'])[center_x_cm].diff() 
            delta_y = df_cm.groupby(df['video_id'])[center_y_cm].diff()
            
            speed_col_name = f'mouse{m}_speed_cm_per_frame'
            df_kinematics.loc[:, speed_col_name] = np.sqrt(delta_x**2 + delta_y**2)
            speed_cols.append(speed_col_name)

    # 2b. Distância Social (Mantida)
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
    
    # --- NOVO: ENGENHARIA DE FEATURES INICIA AQUI ---
    
    # 2c. Features de Postura e Ângulo (Orientação corporal)
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
            # np.arctan2 é robusto e retorna o ângulo correto entre -pi e pi
            df_kinematics.loc[:, angle_col_name] = np.arctan2(dy, dx)
            angle_cols.append(angle_col_name)

    # 2d. Features de Aceleração (Mudança de Velocidade)
    accel_cols = []
    for speed_col in speed_cols: 
        accel_col_name = speed_col.replace('speed', 'accel') 
        # A aceleração é a diferença (derivada) da velocidade
        df_kinematics.loc[:, accel_col_name] = df_kinematics.groupby(df['video_id'])[speed_col].diff()
        accel_cols.append(accel_col_name)

    # 2e. Features de Distância a Pontos Chave (Nariz <-> Base da Cauda/Genitália)
    point_dist_cols = []
    # Cria distâncias Nariz A -> Cauda B e Nariz B -> Cauda A
    for m1, m2 in mouse_pairs: 
        nose1_x, nose1_y = f'mouse{m1}_nose_cm_x', f'mouse{m1}_nose_cm_y'
        tail2_x, tail2_y = f'mouse{m2}_tail_base_cm_x', f'mouse{m2}_tail_base_cm_y'

        nose2_x, nose2_y = f'mouse{m2}_nose_cm_x', f'mouse{m2}_nose_cm_y'
        tail1_x, tail1_y = f'mouse{m1}_tail_base_cm_x', f'mouse{m1}_tail_base_cm_y'
        
        # Distância Nariz M1 -> Base Cauda M2
        if nose1_x in df_cm.columns and tail2_x in df_cm.columns:
            dist_col_name_1_2 = f'dist_nose{m1}_tail{m2}_cm'
            df_kinematics.loc[:, dist_col_name_1_2] = np.sqrt(
                (df_cm[nose1_x] - df_cm[tail2_x])**2 +
                (df_cm[nose1_y] - df_cm[tail2_y])**2
            )
            point_dist_cols.append(dist_col_name_1_2)

        # Distância Nariz M2 -> Base Cauda M1
        if nose2_x in df_cm.columns and tail1_x in df_cm.columns:
            dist_col_name_2_1 = f'dist_nose{m2}_tail{m1}_cm'
            df_kinematics.loc[:, dist_col_name_2_1] = np.sqrt(
                (df_cm[nose2_x] - df_cm[tail1_x])**2 +
                (df_cm[nose2_y] - df_cm[tail1_y])**2
            )
            point_dist_cols.append(dist_col_name_2_1)
            
    # --- FIM DA ENGENHARIA DE FEATURES ---

    # --------------------------------------------------------------------------------
    # 3. Processamento de Metadados (OHE e Normalização)
    # --------------------------------------------------------------------------------
    
    CATEGORICAL_COLS = [
        'arena_type', 'arena_shape', 
        'mouse1_sex', 'mouse2_sex', 'mouse3_sex', 'mouse4_sex',
    ]
    NUMERIC_CONTEXT_COLS = [
        'mouse1_age', 'mouse2_age', 'mouse3_age', 'mouse4_age',
        'arena_width_cm', 'arena_height_cm',
    ]

    df_context = df[CATEGORICAL_COLS + NUMERIC_CONTEXT_COLS].copy()

    # 3a. One-Hot Encoding (OHE)
    df_context = pd.get_dummies(df_context, columns=CATEGORICAL_COLS, dummy_na=False)
    ohe_cols = [col for col in df_context.columns if any(c in col for c in CATEGORICAL_COLS)]
    
    # 3b. Normalização de Variáveis Numéricas (MinMax)
    numeric_context_norm_cols = []
    for col in NUMERIC_CONTEXT_COLS:
        if col in df_context.columns:
            df_context.loc[:, col] = pd.to_numeric(df_context[col], errors='coerce').astype(float)
            df_context.rename(columns={col: f'{col}_norm'}, inplace=True) 
            col_norm = f'{col}_norm'

            min_val = df_context[col_norm].min()
            max_val = df_context[col_norm].max()
            
            if (max_val - min_val) != 0:
                df_context.loc[:, col_norm] = ((df_context[col_norm] - min_val) / (max_val - min_val))
            else:
                df_context.loc[:, col_norm] = 0.0
            
            numeric_context_norm_cols.append(col_norm)
    
    # --------------------------------------------------------------------------------
    # 4. Concatenação Final
    # --------------------------------------------------------------------------------
    
    base_cols = list(set(METADATA_COLUMNS_TO_KEEP_RAW) & set(df.columns))
    df_final = df[base_cols].copy()

    df_final = pd.concat([
        df_final, 
        df_cm, 
        df_kinematics, 
        df_context[ohe_cols + numeric_context_norm_cols]
    ], axis=1)
    
    return df_final.reset_index(drop=True)
# ... (O restante do script if __name__ == "__main__": permanece o mesmo)

# =================================================================
# 2. Execução da Pipeline
# =================================================================

if __name__ == "__main__":
    # ❗ Altere estas pastas 
    INPUT_PATH = Path("MABe-mouse-behavior-detection/processed_videos_final_fixed") 
    OUTPUT_PATH = Path("MABe-mouse-behavior-detection/feature_engineered_data")
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True) # Cria a pasta de saída se não existir
    
    parquet_files = list(INPUT_PATH.rglob("*.parquet"))

    if not parquet_files:
        print(f"❌ NENHUM arquivo Parquet encontrado na pasta de entrada: {INPUT_PATH.absolute()}")
    else:
        print(f"🔍 Encontrados {len(parquet_files)} arquivos para processar.")
        
        # O loop de processamento deve ser lento, monitorado pelo tqdm
        for file_path in tqdm(parquet_files, desc="Processando Features"):
            try:
                # 1. Carrega
                df_raw = pd.read_parquet(file_path, engine='fastparquet')
                
                # 2. Processa
                df_processed = pipeline_feature_engineering(df_raw)
                
                # 3. Salva no novo caminho
                output_file = OUTPUT_PATH / file_path.name
                df_processed.to_parquet(output_file, engine='fastparquet', index=False)
                
            except Exception as e:
                print(f"\n⚠️ ERRO FATAL ao processar {file_path.name}: {e}. Pulando.")
                continue

        print("\n✅ Processamento de Features concluído. Os dados estão prontos na pasta:", OUTPUT_PATH)