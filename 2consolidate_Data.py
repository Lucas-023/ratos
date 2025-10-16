# consolidate_data.py - Versão Ultraestável (Memmap + Gravação Segura + Limpeza de NaN)

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Any
import os 

# =========================================================
# CONFIGURAÇÕES E PARÂMETROS
# =========================================================

# 🚨 AJUSTE AQUI: Caminho para os arquivos Parquet *PROCESSADOS* de TREINO 🚨
BASE_PATH_TRAIN = Path("MABe-mouse-behavior-detection/processed_videos_final_fixed") 

OUTPUT_X = "consolidated_X.npy"    
OUTPUT_Y = "consolidated_Y.csv"    
X_MEAN_PATH = "X_mean.npy"         
X_STD_PATH = "X_std.npy"           

# --- DEFINIÇÕES DE COLUNAS ---
try:
    from dataloader import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    print("⚠️ Aviso: dataloader.py não encontrado. Usando definições mockadas.")
    FEATURE_COLUMNS = [f'x_{i}' for i in range(1, 118)]
    TARGET_COLUMN = 'behavior'

N_FEATURES = len(FEATURE_COLUMNS)
parquet_files = list(BASE_PATH_TRAIN.rglob("*.parquet"))

if not parquet_files:
    print(f"❌ NENHUM arquivo Parquet encontrado em {BASE_PATH_TRAIN.resolve()}.")
    exit()

# =========================================================
# FUNÇÃO AUXILIAR: Extração Segura de Labels
# =========================================================

def safe_extract_labels(label_raw: Any) -> List[str]:
    """
    Trata formatos de labels e retorna uma lista de labels válidas.
    """
    if isinstance(label_raw, (list, np.ndarray, pd.Series)):
         return [str(l).strip() for l in label_raw if str(l).strip()]
    if pd.isna(label_raw): 
        return []
    label_str = str(label_raw).strip()
    if not label_str or label_str.lower() in ('nan', '0.0', '0'):
        return []
    return [l.strip() for l in label_str.split(';') if l.strip()]


# =========================================================
# 1. PRÉ-CÁLCULO DO TAMANHO TOTAL (PASSAGEM 0)
# =========================================================

total_frames = 0
print("🔍 Calculando o número total de frames em todos os arquivos...")
for file_path in tqdm(parquet_files, desc="Contando Frames"):
    try:
        df_temp = pd.read_parquet(file_path, engine='fastparquet', columns=[TARGET_COLUMN])
        total_frames += df_temp.shape[0]
    except Exception as e:
        print(f"\n⚠️ ERRO ao contar frames em {file_path.name}: {e}. Pulando.")

if total_frames == 0:
    print("Nenhum frame válido encontrado para consolidação. Saindo.")
    exit()

print(f"✅ Total de frames a serem consolidados: {total_frames}")


# =========================================================
# 2. CÁLCULO DE ESTATÍSTICAS (PRIMEIRA PASSAGEM)
# =========================================================

sum_x = np.zeros(N_FEATURES, dtype=np.float64)
sum_x_sq = np.zeros(N_FEATURES, dtype=np.float64)
total_frames_count = 0 

print("\n📊 Iniciando a PRIMEIRA PASSAGEM: Cálculo de Média e Variância...")
for file_path in tqdm(parquet_files, desc="Passagem 1/2: Calculando Estatísticas"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        features_df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
        features_np = features_df.values.astype(np.float64) 
        
        sum_x += np.sum(features_np, axis=0)
        sum_x_sq += np.sum(features_np**2, axis=0)
        total_frames_count += len(features_np)
        
    except Exception as e:
        print(f"\n⚠️ ERRO no cálculo de estatísticas em {file_path.name}: {e}. Pulando.")
        continue

# --- Cálculo Final do Z-Score ---
X_mean = sum_x / total_frames_count
X_var = (sum_x_sq / total_frames_count) - (X_mean**2)
X_std = np.sqrt(X_var)

# CORREÇÃO CRÍTICA DE ESTABILIDADE DO STD
EPSILON = 1e-8
X_std[X_std < EPSILON] = EPSILON 
print(f"   STD MÍNIMO (Após correção): {np.min(X_std):.1e}")

np.save(X_MEAN_PATH, X_mean.astype(np.float32))
np.save(X_STD_PATH, X_std.astype(np.float32))

print(f"✅ Estatísticas calculadas e salvas em {X_MEAN_PATH} e {X_STD_PATH}.")


# =========================================================
# 3. CONSOLIDAÇÃO, NORMALIZAÇÃO E ESCRITA (SEGUNDA PASSAGEM - MEMMAP SEGURO)
# =========================================================

# PRÉ-ALOCAÇÃO COM MEMMAP (Baixo uso de RAM durante a escrita)
print(f"\nPré-alocando arquivo temporário ({total_frames} linhas x {N_FEATURES} features) via Memmap...")
TEMP_X_PATH = "temp_" + OUTPUT_X
X_memmap_temp = np.memmap(
    TEMP_X_PATH, 
    dtype=np.float32, 
    mode='w+', 
    shape=(total_frames, N_FEATURES)
)

# Reseta o arquivo CSV de labels e escreve o cabeçalho
with open(OUTPUT_Y, 'w', encoding='utf-8') as f_csv:
    f_csv.write(f"{TARGET_COLUMN}\n")

current_index = 0
print("🚀 Iniciando a SEGUNDA PASSAGEM: Normalização e Gravação no Memmap...")

for file_path in tqdm(parquet_files, desc="Passagem 2/2: Gravando Normalizado"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        n_rows = len(df)
        
        # --- A) Gravação das FEATURES (X) ---
        features_df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
        features_np = features_df.values.astype(np.float32)
        
        # NORMALIZAÇÃO Z-SCORE 
        features_normalized = (features_np - X_mean.astype(np.float32)) / X_std.astype(np.float32)
        
        # 🚨 CORREÇÃO CRÍTICA DE NAN (IMPUTAÇÃO) 🚨
        # Substitui qualquer NaN gerado (ou herdado) por 0.0
        features_normalized[np.isnan(features_normalized)] = 0.0
        
        # Escreve o bloco de dados NORMALIZADO no memmap temporário
        X_memmap_temp[current_index : current_index + n_rows] = features_normalized
        
        # --- B) Gravação dos LABELS (Y) ---
        labels_series = df[TARGET_COLUMN].apply(
            lambda x: ";".join(safe_extract_labels(x))
        )
        labels_series.to_csv(OUTPUT_Y, mode='a', index=False, header=False, encoding='utf-8')
        
        current_index += n_rows
        
    except Exception as e:
        print(f"\n⚠️ ERRO (PULANDO ARQUIVO) ao normalizar e gravar {file_path.name}: {e}. A consolidação continua.")
        continue 

# Garante que todos os dados tenham sido escritos no disco
X_memmap_temp.flush() 

# 🚨 ETAPA CRÍTICA DE CORREÇÃO DO FORMATO 🚨
print("\n💾 SALVAMENTO SEGURO: Carregando (última vez) e re-salvando o Memmap com np.save...")
try:
    # Carrega o array inteiro na RAM (Se 32GB aguentar, funcionará)
    X_final_array = np.array(X_memmap_temp) 
    
    # Salva o array de forma segura com np.save padrão
    np.save(OUTPUT_X, X_final_array)
    
    # Remove o arquivo temporário
    os.remove(TEMP_X_PATH)

    print("\n----------------------------------------------------")
    print(f"✅ CONSOLIDAÇÃO CONCLUÍDA. O arquivo final {OUTPUT_X} foi salvo de forma segura.")
    print("----------------------------------------------------")

except Exception as e:
    print(f"\n❌ ERRO FATAL na re-gravação: {e}. O problema é de memória/sistema operacional.")
    print("O arquivo FINAL NÃO foi criado. Tente criar um novo ambiente Python.")