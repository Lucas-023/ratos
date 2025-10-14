# consolidate_test_data.py - Consolidação Robusta para os Dados de TESTE

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Importa definições do seu dataloader original (necessário para colunas)
# Assumo que estas definições estão em dataloader.py
try:
    from dataloader import FEATURE_COLUMNS, TARGET_COLUMN, safe_extract_labels
except ImportError:
    # Caso não seja possível importar, usamos as definições padrão
    print("⚠️ Aviso: dataloader.py não encontrado. Usando definições de coluna mockadas.")
    FEATURE_COLUMNS = [f'x_{i}_body_center' for i in range(1, 5)] + ['vel_x_1_body_center', 'total_movement_1_body_center']
    TARGET_COLUMN = 'behavior' 
    def safe_extract_labels(x):
        return x.split(';') if isinstance(x, str) else []


# =========================================================
# CONFIGURAÇÕES E PARÂMETROS
# =========================================================
# 🚨 AJUSTE AQUI: Caminho para os arquivos Parquet *PROCESSADOS* de TESTE 🚨
BASE_PATH_TEST = Path("MABe-mouse-behavior-detection/test_tracking/processed_videos_test_fixed") 

OUTPUT_X = "consolidated_TEST_X.npy" # Features de TESTE
OUTPUT_Y = "consolidated_TEST_Y.csv" # Labels de TESTE
N_FEATURES = len(FEATURE_COLUMNS)

parquet_files = list(BASE_PATH_TEST.rglob("*.parquet"))

if not parquet_files:
    print(f"❌ NENHUM arquivo Parquet encontrado em {BASE_PATH_TEST.resolve()}.")
    exit()

# =========================================================
# 1. PRÉ-CÁLCULO DO TAMANHO TOTAL (Contando por FRAME)
# =========================================================

total_frames = 0
print("🔍 Calculando o número total de frames em todos os arquivos de TESTE...")
for file_path in tqdm(parquet_files, desc="Contando Frames de Teste"):
    try:
        # CORREÇÃO: Conta pelo 'frame' para evitar falha se 'behavior' estiver ausente
        df_temp = pd.read_parquet(file_path, engine='fastparquet', columns=['frame'])
        total_frames += df_temp.shape[0]
    except Exception as e:
        print(f"\n⚠️ ERRO ao contar frames em {file_path.name}: {e}. Pulando.")

if total_frames == 0:
    print("Nenhum frame válido encontrado para consolidação de TESTE. Saindo.")
    exit()

print(f"✅ Total de frames de TESTE a serem consolidados: {total_frames}")

# =========================================================
# 2. PRÉ-ALOCAÇÃO DO ARQUIVO .NPY (USANDO MEMMAP)
# =========================================================

print(f"Pré-alocando {OUTPUT_X} ({total_frames} linhas x {N_FEATURES} features)...")
X_memmap = np.memmap(
    OUTPUT_X, 
    dtype=np.float32, 
    mode='w+', 
    shape=(total_frames, N_FEATURES)
)

# =========================================================
# 3. CONSOLIDAÇÃO INCREMENTAL ROBUSTA
# =========================================================

current_index = 0
# Abre o arquivo CSV de labels de teste no modo 'w' (write) e escreve o cabeçalho
with open(OUTPUT_Y, 'w') as f_csv:
    f_csv.write(f"{TARGET_COLUMN}\n")

print("Iniciando escrita incremental dos dados de TESTE com checagem de colunas...")
for file_path in tqdm(parquet_files, desc="Gravando Features e Labels de Teste"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        n_rows = len(df)
        
        # --- A) Gravação das FEATURES (X) - Correção de Schema Mismatch ---
        existing_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        features_df = df[existing_features]
        # Adiciona colunas ausentes (features de velocidade, etc.) com fill_value=0.0
        features_df = features_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

        features_np = features_df.values.astype(np.float32)
        X_memmap[current_index : current_index + n_rows] = features_np
        
        # --- B) Gravação dos LABELS (Y) ---
        # Garante a coluna TARGET_COLUMN (que agora é behavior)
        if TARGET_COLUMN not in df.columns:
             df[TARGET_COLUMN] = 'nan' # Se a coluna foi perdida por algum motivo, trata aqui
             
        # O safe_extract_labels é usado para normalizar labels, mas para teste com 'nan', 
        # garantimos que 'nan' seja escrito diretamente.
        labels_series = df[TARGET_COLUMN].apply(
            lambda x: ";".join(safe_extract_labels(x)) if x != 'nan' and pd.notna(x) else 'nan'
        )
        labels_series.to_csv(
            OUTPUT_Y, 
            mode='a', 
            index=False, 
            header=False,
            encoding='utf-8'
        )
        
        current_index += n_rows
        
    except Exception as e:
        print(f"\n⚠️ ERRO (PULANDO ARQUIVO DE TESTE) ao processar {file_path.name}: {e}. A consolidação continua.")
        continue 

X_memmap.flush() 

print("\n🚀 Consolidação de TESTE concluída!")
print(f"Arquivos prontos: {OUTPUT_X} e {OUTPUT_Y}")