# consolidate_data.py - Versão Robusta para Inconsistência de Colunas (Schema Mismatch)
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Importa definições do seu dataloader original
from dataloader import FEATURE_COLUMNS, TARGET_COLUMN, safe_extract_labels

# =========================================================
# CONFIGURAÇÕES E PARÂMETROS
# =========================================================
BASE_PATH = Path("MABe-mouse-behavior-detection/feature_engineered_data")
OUTPUT_X = "consolidated_X.npy" # Features (117 colunas, float32)
OUTPUT_Y = "consolidated_Y.csv" # Labels (comportamentos, string)
N_FEATURES = len(FEATURE_COLUMNS)

parquet_files = list(BASE_PATH.rglob("*.parquet"))

if not parquet_files:
    print(f"❌ NENHUM arquivo Parquet encontrado em {BASE_PATH.resolve()}.")
    exit()

# =========================================================
# 1. PRÉ-CÁLCULO DO TAMANHO TOTAL
# =========================================================

total_frames = 0
print("🔍 Calculando o número total de frames...")
for file_path in tqdm(parquet_files, desc="Contando Frames"):
    try:
        # Lê apenas a coluna alvo para contar as linhas
        df_temp = pd.read_parquet(file_path, engine='fastparquet', columns=[TARGET_COLUMN])
        total_frames += df_temp.shape[0]
    except Exception as e:
        print(f"\n⚠️ ERRO ao contar frames em {file_path.name}: {e}. Pulando.")

if total_frames == 0:
    print("Nenhum frame válido encontrado para consolidação. Saindo.")
    exit()

print(f"✅ Total de frames a serem consolidados: {total_frames}")

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
# Abre o arquivo CSV de labels no modo 'w' (write) e escreve o cabeçalho
with open(OUTPUT_Y, 'w') as f_csv:
    f_csv.write(f"{TARGET_COLUMN}\n")

print("Iniciando escrita incremental com checagem de colunas...")
for file_path in tqdm(parquet_files, desc="Gravando Features e Labels"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        n_rows = len(df)
        
        # Checagem de integridade mínima
        if TARGET_COLUMN not in df.columns or n_rows == 0:
            raise ValueError(f"Target '{TARGET_COLUMN}' ausente ou arquivo vazio.")
        
        # --- A) Gravação das FEATURES (X) - CORREÇÃO CRÍTICA DO ERRO 'not in index' ---
        
        # 1. Seleciona as colunas de features que REALMENTE existem no DataFrame
        existing_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        features_df = df[existing_features]
        
        # 2. REINDEXAÇÃO: Reorganiza o DF para ter TODAS as colunas de FEATURE_COLUMNS,
        #    preenchendo as que faltam (os mouses não rastreados) com 0.0.
        features_df = features_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

        # Converte para array float32
        features_np = features_df.values.astype(np.float32)
        
        # Escreve o bloco de dados diretamente no memmap
        X_memmap[current_index : current_index + n_rows] = features_np
        
        # --- B) Gravação dos LABELS (Y) ---
        
        # Processa os labels (converte a lista/array em string separada por ';')
        labels_series = df[TARGET_COLUMN].apply(
            lambda x: ";".join(safe_extract_labels(x))
        )
        
        # Escreve os labels diretamente no arquivo CSV (append)
        labels_series.to_csv(
            OUTPUT_Y, 
            mode='a', 
            index=False, 
            header=False,
            encoding='utf-8'
        )
        
        # Atualiza o índice para o próximo bloco
        current_index += n_rows
        
    except Exception as e:
        print(f"\n⚠️ ERRO (PULANDO ARQUIVO) ao processar {file_path.name}: {e}. A consolidação continua.")
        continue 

# Força a escrita de qualquer dado em cache para o disco
X_memmap.flush() 

print("\n🚀 Consolidação concluída! Arquivos 'consolidated_X.npy' e 'consolidated_Y.csv' prontos.")