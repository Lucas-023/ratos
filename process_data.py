# process_data.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =================================================================
# 1. Definição da Pipeline de Feature Engineering
# =================================================================

# Colunas de metadados que queremos manter no arquivo final (além das features)
METADATA_COLUMNS = [
    'frame', 'behavior', 'lab_id', 'video_id', 'mouse1_strain', 'mouse1_color',
    'mouse1_sex', 'mouse1_id', 'mouse1_age', 'mouse1_condition', 
    'mouse2_strain', 'mouse2_color', 'mouse2_sex', 'mouse2_id', 'mouse2_age', 
    'mouse2_condition', 'mouse3_strain', 'mouse3_color', 'mouse3_sex', 
    'mouse3_id', 'mouse3_age', 'mouse3_condition', 'mouse4_strain', 
    'mouse4_color', 'mouse4_sex', 'mouse4_id', 'mouse4_age', 'mouse4_condition', 
    'frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 
    'video_width_pix', 'video_height_pix', 'arena_width_cm', 'arena_height_cm', 
    'arena_shape', 'arena_type', 'body_parts_tracked', 'behaviors_labeled', 
    'tracking_method', 'unique_frame_id'
]

def pipeline_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica o feature engineering completo a um ÚNICO arquivo Parquet."""
    
    if df.empty:
        return df

    # Encontra TODAS as colunas de coordenadas (_x, _y)
    coord_cols = [col for col in df.columns 
                  if (col.endswith('_x') or col.endswith('_y')) and col.startswith('mouse')]

    # 2a. Padroniza 0.0 como NaN nas coordenadas (Assume que 0.0 significa 'não detectado')
    df.loc[:, coord_cols] = df.loc[:, coord_cols].replace(0.0, np.nan)

    # 2b. Interpolação Linear (A interpolação deve ser POR VÍDEO, mesmo que o DF seja de um único vídeo)
    for col in coord_cols:
         # Usamos .copy() para evitar SettingWithCopyWarning, pois estamos modificando o DF
        df.loc[:, col] = df.groupby('video_id')[col].transform(
             lambda x: x.interpolate(method='linear', limit=10, limit_direction='both')
        ).copy()

    # 3. Normalização: Pixels para Centímetros (CM)
    for col in coord_cols:
        col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
        df.loc[:, col_cm] = df[col] / df['pix_per_cm_approx']

    # 4. Criação de Features de Movimento (Velocidade)
    for m in range(1, 5):
        center_x_cm = f'mouse{m}_body_center_cm_x'
        center_y_cm = f'mouse{m}_body_center_cm_y'
        
        # Calcula a diferença de X e Y entre frames
        df.loc[:, f'mouse{m}_delta_x'] = df.groupby('video_id')[center_x_cm].diff().copy()
        df.loc[:, f'mouse{m}_delta_y'] = df.groupby('video_id')[center_y_cm].diff().copy()
        
        # Calcula a velocidade (distância euclidiana da mudança: sqrt(dx² + dy²))
        df.loc[:, f'mouse{m}_speed_cm_per_frame'] = np.sqrt(
            df[f'mouse{m}_delta_x']**2 + df[f'mouse{m}_delta_y']**2
        ).copy()

    # 5. Criação de Features de Interação (Distância entre Ratos)
    # Calcule todas as 6 combinações de pares: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for m1, m2 in mouse_pairs:
        center1_x = f'mouse{m1}_body_center_cm_x'
        center1_y = f'mouse{m1}_body_center_cm_y'
        center2_x = f'mouse{m2}_body_center_cm_x'
        center2_y = f'mouse{m2}_body_center_cm_y'

        df.loc[:, f'dist_m{m1}_m{m2}_cm'] = np.sqrt(
            (df[center1_x] - df[center2_x])**2 +
            (df[center1_y] - df[center2_y])**2
        )
    
    # Seleciona apenas as colunas úteis para o modelo (CM coords, Metadados e Novas Features)
    cm_cols = [col for col in df.columns if col.endswith('_cm_x') or col.endswith('_cm_y')]
    speed_cols = [col for col in df.columns if col.endswith('_speed_cm_per_frame')]
    dist_cols = [col for col in df.columns if col.startswith('dist_m') and col.endswith('_cm')]
    
    # As colunas delta_x e delta_y são intermediárias, não as incluímos no final
    
    final_cols = METADATA_COLUMNS + cm_cols + speed_cols + dist_cols
    
    # Garante que o DataFrame final só tenha as colunas que queremos, ignorando NaNs (que serão tratados mais tarde)
    df_processed = df[list(set(final_cols) & set(df.columns))]
    
    return df_processed.reset_index(drop=True)

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