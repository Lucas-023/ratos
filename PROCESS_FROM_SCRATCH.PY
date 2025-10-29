# process_from_scratch.py
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math

# ===================================================================
# 1. CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS DADOS GLOBAIS
# ===================================================================
print("Iniciando o pipeline de processamento de dados do zero.")

# 🚨 AJUSTE AQUI: Caminhos para as pastas de dados baixados do Kaggle 🚨
BASE_PATH = Path("./MABe-mouse-behavior-detection") # Pasta principal da competição
TRAIN_TRACKING_PATH = BASE_PATH / "train_tracking"
TRAIN_ANNOTATIONS_PATH = BASE_PATH / "train_annotations.parquet"
SEQUENCE_METADATA_PATH = BASE_PATH / "sequence_metadata.csv"

# 🚨 AJUSTE AQUI: Pasta onde os arquivos processados serão salvos 🚨
OUTPUT_PROCESSED_PATH = BASE_PATH / "train_processed"
OUTPUT_PROCESSED_PATH.mkdir(exist_ok=True) # Cria a pasta de saída

# --- Carregando arquivos de metadados ---
try:
    print("Carregando metadados e anotações...")
    df_meta = pd.read_csv(SEQUENCE_METADATA_PATH)
    df_annotations = pd.read_parquet(TRAIN_ANNOTATIONS_PATH)
    
    # Junta os dois para ter um mapa completo
    df_master = pd.merge(df_meta, df_annotations, on='sequence_id')
    print("✅ Metadados e anotações carregados com sucesso.")
    
except FileNotFoundError as e:
    print(f"❌ ERRO: Arquivo não encontrado: {e}. Verifique os caminhos em BASE_PATH.")
    exit()

# Extrai uma lista de todos os arquivos de tracking
all_tracking_files = list(TRAIN_TRACKING_PATH.rglob("*.parquet"))
print(f"Encontrados {len(all_tracking_files)} arquivos de tracking para processar.")


# ===================================================================
# 2. FUNÇÃO PRINCIPAL DE ENGENHARIA DE FEATURES (POR ARQUIVO)
# ===================================================================

def process_single_tracking_file(file_path: Path, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa um único arquivo de tracking, adiciona anotações e cria features egocêntricas.
    Esta é a técnica mais importante vista nos notebooks de sucesso.
    """
    sequence_id = int(file_path.stem)
    
    try:
        # Lê o arquivo de tracking usando o motor 'fastparquet' para robustez
        df_tracking = pd.read_parquet(file_path, engine='fastparquet')
    except Exception:
        # Se falhar, retorna um DataFrame vazio para ser ignorado
        return pd.DataFrame()

    # Pega as informações relevantes para esta sequência do DataFrame mestre
    sequence_info = master_df[master_df['sequence_id'] == sequence_id].copy()
    
    # Adiciona as anotações de comportamento a cada frame do tracking
    df_processed = pd.merge(df_tracking, sequence_info[['frame', 'behavior']], on='frame', how='left')
    
    # Preenche os labels para frente (se um comportamento dura vários frames, ele só é anotado no primeiro)
    df_processed['behavior'] = df_processed['behavior'].ffill()
    # Preenche qualquer NaN restante com "no_behavior"
    df_processed['behavior'] = df_processed['behavior'].fillna('no_behavior')

    # --- Engenharia de Features Egocêntricas ---
    # A ideia é transformar o sistema de coordenadas para que o "rato focal" (mouse1)
    # esteja sempre na origem (0,0) e olhando para "cima". Isso torna os movimentos
    # dos outros ratos relativos e mais fáceis para o modelo aprender.
    
    # Pontos de referência para o rato focal (mouse1)
    focal_x = df_processed['mouse1_nose_x'].values
    focal_y = df_processed['mouse1_nose_y'].values
    focal_neck_x = df_processed['mouse1_neck_x'].values
    focal_neck_y = df_processed['mouse1_neck_y'].values

    # Calcula o ângulo da cabeça do rato focal
    # A função atan2 é usada para obter o ângulo correto em radianos
    angles = np.arctan2(focal_y - focal_neck_y, focal_x - focal_neck_x)
    
    cos_angles = np.cos(-angles)
    sin_angles = np.sin(-angles)

    # Itera sobre todos os ratos e todas as partes do corpo para aplicar a transformação
    for m in range(1, 5):  # mouse1, mouse2, mouse3, mouse4
        for part in ['nose', 'ear_left', 'ear_right', 'neck', 'body_center', 'tail_base']:
            px = df_processed[f'mouse{m}_{part}_x'].values
            py = df_processed[f'mouse{m}_{part}_y'].values
            
            # 1. Translação: Move o sistema para que o pescoço do rato focal seja a origem
            translated_x = px - focal_neck_x
            translated_y = py - focal_neck_y

            # 2. Rotação: Gira o sistema para que o rato focal olhe para "cima"
            rotated_x = translated_x * cos_angles - translated_y * sin_angles
            rotated_y = translated_x * sin_angles + translated_y * cos_angles
            
            # Salva as novas coordenadas egocêntricas
            df_processed[f'ego_mouse{m}_{part}_x'] = rotated_x
            df_processed[f'ego_mouse{m}_{part}_y'] = rotated_y
            
    return df_processed

# ===================================================================
# 3. EXECUÇÃO DO PROCESSAMENTO EM LOTE
# ===================================================================
print("\n🚀 Iniciando o processamento de todos os arquivos. Isso pode demorar...")

# Lista das colunas de features egocêntricas que queremos salvar, mais o label
ego_features = []
for m in range(1, 5):
    for part in ['nose', 'ear_left', 'ear_right', 'neck', 'body_center', 'tail_base']:
        ego_features.append(f'ego_mouse{m}_{part}_x')
        ego_features.append(f'ego_mouse{m}_{part}_y')
        
columns_to_save = ego_features + ['behavior']

processed_count = 0
for file_path in tqdm(all_tracking_files, desc="Processando e Salvando Arquivos"):
    # Processa o arquivo
    df_final = process_single_tracking_file(file_path, df_master)
    
    # Se o processamento foi bem-sucedido e o DataFrame não está vazio
    if not df_final.empty:
        # Seleciona apenas as colunas que nos interessam
        df_to_save = df_final[columns_to_save]
        
        # Cria o caminho de saída
        output_file = OUTPUT_PROCESSED_PATH / file_path.name
        
        # Salva o arquivo processado, usando um motor robusto
        df_to_save.to_parquet(output_file, engine='fastparquet')
        processed_count += 1

print("\n----------------------------------------------------")
print(f"✅ Processamento concluído!")
print(f"{processed_count} de {len(all_tracking_files)} arquivos foram processados e salvos com sucesso.")
print(f"Os dados processados estão na pasta: {OUTPUT_PROCESSED_PATH.resolve()}")
print("----------------------------------------------------")
print("\nPróximo passo: Use os arquivos desta pasta para treinar seu modelo.")