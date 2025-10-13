# process_data.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# ... (o resto dos seus imports)


# =================================================================
# 1. Definição da Pipeline de Feature Engineering (Função SUBSTITUÍDA)
# =================================================================

# Colunas de metadados BRUTOS que queremos manter no arquivo final (além das features)
# NOTA: Removemos daqui as colunas que serão OHE ou normalizadas!
METADATA_COLUMNS_TO_KEEP_RAW = [
    'frame', 'behavior', 'video_id', 'unique_frame_id',
    'lab_id', 
    # Mantemos algumas informações estáticas que não serão OHE/normalizadas
    'frames_per_second', 'video_duration_sec', 'pix_per_cm_approx', 
    'video_width_pix', 'video_height_pix', 
    'body_parts_tracked', 'behaviors_labeled', 'tracking_method',
    # Mantemos strain, color, condition, id (você pode remover se quiser manter o dataset mais limpo)
    'mouse1_strain', 'mouse1_color', 'mouse1_condition', 'mouse1_id',
    'mouse2_strain', 'mouse2_color', 'mouse2_condition', 'mouse2_id',
    'mouse3_strain', 'mouse3_color', 'mouse3_condition', 'mouse3_id',
    'mouse4_strain', 'mouse4_color', 'mouse4_condition', 'mouse4_id',
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

    # 2b. Interpolação Linear 
    for col in coord_cols:
        df.loc[:, col] = df.groupby('video_id')[col].transform(
             lambda x: x.interpolate(method='linear', limit=10, limit_direction='both')
        ).copy()

    # 3. Normalização: Pixels para Centímetros (CM)
    for col in coord_cols:
        col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
        df.loc[:, col_cm] = df[col] / df['pix_per_cm_approx']
    
    # Lista as novas colunas CM criadas
    cm_cols = [col for col in df.columns if col.endswith('_cm_x') or col.endswith('_cm_y')]


    # 4. Criação de Features de Movimento (Velocidade)
    for m in range(1, 5):
        center_x_cm = f'mouse{m}_body_center_cm_x'
        center_y_cm = f'mouse{m}_body_center_cm_y'
        
        # Calcula a diferença de X e Y entre frames
        df.loc[:, f'mouse{m}_delta_x'] = df.groupby('video_id')[center_x_cm].diff().copy()
        df.loc[:, f'mouse{m}_delta_y'] = df.groupby('video_id')[center_y_cm].diff().copy()
        
        # Calcula a velocidade 
        df.loc[:, f'mouse{m}_speed_cm_per_frame'] = np.sqrt(
            df[f'mouse{m}_delta_x']**2 + df[f'mouse{m}_delta_y']**2
        ).copy()

    # Lista as novas colunas de velocidade
    speed_cols = [f'mouse{m}_speed_cm_per_frame' for m in range(1, 5)]


    # 5. Criação de Features de Interação (Distância entre Ratos)
    mouse_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    dist_cols = []
    for m1, m2 in mouse_pairs:
        dist_col_name = f'dist_m{m1}_m{m2}_cm'
        center1_x = f'mouse{m1}_body_center_cm_x'
        center1_y = f'mouse{m1}_body_center_cm_y'
        center2_x = f'mouse{m2}_body_center_cm_x'
        center2_y = f'mouse{m2}_body_center_cm_y'

        df.loc[:, dist_col_name] = np.sqrt(
            (df[center1_x] - df[center2_x])**2 +
            (df[center1_y] - df[center2_y])**2
        )
        dist_cols.append(dist_col_name)

    
    # --------------------------------------------------------------------------------
    # 6. ❗ INCLUSÃO E PROCESSAMENTO DE METADADOS (OHE e Normalização) ❗
    # --------------------------------------------------------------------------------
    
    # Colunas Categóricas para One-Hot Encoding (OHE)
    CATEGORICAL_COLS = [
        'arena_type', 'arena_shape', 
        'mouse1_sex', 'mouse2_sex', 'mouse3_sex', 'mouse4_sex',
    ]

    # Colunas Numéricas Contextuais para Normalização (Idade e Dimensões da Arena)
    NUMERIC_CONTEXT_COLS = [
        'mouse1_age', 'mouse2_age', 'mouse3_age', 'mouse4_age',
        'arena_width_cm', 'arena_height_cm',
    ]
    
    # --- 6a. One-Hot Encoding (OHE) ---
    # Aplica OHE e adiciona as novas colunas (ex: 'arena_type_OpenField')
    # Nota: get_dummies lida bem com NaNs, mas 'dummy_na=True' criaria uma coluna '_nan'
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dummy_na=False)
    
    # Encontra as colunas OHE que foram criadas
    ohe_cols = [col for col in df.columns if any(c in col for c in CATEGORICAL_COLS)]
    
    # --- 6b. Normalização de Variáveis Numéricas ---
    for col in NUMERIC_CONTEXT_COLS:
        if col in df.columns:
            # Garante que a coluna não será incluída no final com o nome antigo
            df.rename(columns={col: f'{col}_norm'}, inplace=True) 
            col_norm = f'{col}_norm'

            min_val = df[col_norm].min()
            max_val = df[col_norm].max()
            
            if (max_val - min_val) != 0:
                # Aplica MinMax Scaler
                df.loc[:, col_norm] = ((df[col_norm] - min_val) / (max_val - min_val)).copy()
            else:
                # Caso todos os valores sejam iguais, define como 0
                df.loc[:, col_norm] = 0.0

    # Lista as novas colunas numéricas normalizadas
    numeric_context_norm_cols = [f'{col}_norm' for col in NUMERIC_CONTEXT_COLS if f'{col}_norm' in df.columns]

    # --------------------------------------------------------------------------------
    # 7. ❗ SELEÇÃO FINAL DE COLUNAS ❗
    # --------------------------------------------------------------------------------

    final_cols = (
        METADATA_COLUMNS_TO_KEEP_RAW + # Metadados estáticos (frame, behavior, id, etc.)
        cm_cols +                      # Coordenadas Normalizadas (CM)
        speed_cols +                   # Velocidade
        dist_cols +                    # Distância Social
        ohe_cols +                     # Metadados Categóricos OHE
        numeric_context_norm_cols      # Metadados Numéricos Normalizados
    )
    
    # Garante que as colunas existem e remove duplicatas acidentais
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