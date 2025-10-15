# consolidate_data.py - Consolidação de Dados de Treino/Validação

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

# =========================================================
# CONFIGURAÇÕES E PARÂMETROS
# =========================================================

# 🚨 AJUSTE AQUI: Caminho para os arquivos Parquet *PROCESSADOS* de TREINO 🚨
# Assumindo que você tem uma pasta separada para os dados de treino processados
BASE_PATH_TRAIN = Path("MABe-mouse-behavior-detection/processed_videos_final_fixed") 

OUTPUT_X = "consolidated_X.npy"  # Features de TREINO/VAL
OUTPUT_Y = "consolidated_Y.csv"  # Labels de TREINO/VAL

# Importa as definições do seu dataloader original
try:
    # Estas colunas devem ser as 117 features geradas pelo processamento
    from dataloader import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    print("⚠️ Aviso: dataloader.py não encontrado. Usando definições mockadas.")
    # Usando colunas mockadas (AJUSTE ISSO se precisar)
    FEATURE_COLUMNS = [f'x_{i}_body_center' for i in range(1, 5)] + ['vel_x_1_body_center', 'total_movement_1_body_center']
    TARGET_COLUMN = 'behavior'

N_FEATURES = len(FEATURE_COLUMNS)

# Lista de todos os arquivos Parquet que serão consolidados
parquet_files = list(BASE_PATH_TRAIN.rglob("*.parquet"))

if not parquet_files:
    print(f"❌ NENHUM arquivo Parquet encontrado em {BASE_PATH_TRAIN.resolve()}.")
    exit()

# Função auxiliar para extrair labels corretamente (Multi-Label)
# NO ARQUIVO consolidate_data.py
# Substitua a função original por esta:

from typing import Any # Adicione no topo se ainda não tiver

def safe_extract_labels(label_raw: Any) -> List[str]:
    """
    Trata formatos de labels, sendo robusta contra o erro 'ambiguous truth value'.
    Retorna uma lista de strings de labels válidas.
    """
    
    # 1. Trata objetos não escalares (Arrays, Listas, Series), que causam o erro.
    if isinstance(label_raw, (list, np.ndarray, pd.Series)):
         # Converte elementos para string e filtra valores vazios.
         return [str(l).strip() for l in label_raw if str(l).strip()]
         
    # 2. Trata valores NaN ou nulos (escalares)
    # pd.isna() funciona corretamente com valores escalares.
    if pd.isna(label_raw): 
        return []
    
    # 3. Trata strings escalares (strings normais, incluindo a string 'nan')
    label_str = str(label_raw).strip()
    
    if not label_str or label_str.lower() == 'nan':
        return []
        
    # 4. Processa a string multi-label (ex: 'sniff;grooming')
    # O if l.strip() garante que strings vazias entre delimitadores sejam ignoradas.
    return [l.strip() for l in label_str.split(';') if l.strip()]

# O restante do seu código (a chamada .apply(lambda x: ";".join(safe_extract_labels(x))))
# continuará a funcionar com esta nova função, mas sem o erro.


# =========================================================
# 1. PRÉ-CÁLCULO DO TAMANHO TOTAL
# =========================================================

total_frames = 0
print("🔍 Calculando o número total de frames em todos os arquivos...")
for file_path in tqdm(parquet_files, desc="Contando Frames"):
    try:
        # Lê apenas a coluna TARGET_COLUMN (que deve ser 'behavior') para contar as linhas
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
# 'w+' permite escrever e ler o arquivo; ele será criado se não existir
X_memmap = np.memmap(
    OUTPUT_X, 
    dtype=np.float32, 
    mode='w+', 
    shape=(total_frames, N_FEATURES)
)

# =========================================================
# 3. CONSOLIDAÇÃO INCREMENTAL
# =========================================================

current_index = 0
# Abre o arquivo CSV de labels no modo 'w' (write) e escreve o cabeçalho
with open(OUTPUT_Y, 'w', encoding='utf-8') as f_csv:
    f_csv.write(f"{TARGET_COLUMN}\n")

print("Iniciando escrita incremental dos dados...")
for file_path in tqdm(parquet_files, desc="Gravando Features e Labels"):
    try:
        # Lê todas as colunas do Parquet processado
        df = pd.read_parquet(file_path, engine='fastparquet')
        n_rows = len(df)
        
        # --- A) Gravação das FEATURES (X) ---
        
        # Filtra as colunas de feature que existem no DF atual (robustez)
        existing_features = [col for col in FEATURE_COLUMNS if col in df.columns]
        features_df = df[existing_features]
        
        # Adiciona colunas ausentes (fill_value=0.0) e garante a ordem
        features_df = features_df.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)

        # Converte para NumPy e garante o dtype correto (np.float32)
        features_np = features_df.values.astype(np.float32)
        
        # Escreve o bloco de dados no memmap
        X_memmap[current_index : current_index + n_rows] = features_np
        
        # --- B) Gravação dos LABELS (Y) ---
        
        # Extrai a coluna de labels (TARGET_COLUMN)
        labels_series = df[TARGET_COLUMN].apply(
            # Aplica a função de limpeza e reconstrói a string 'comp1;comp2'
            lambda x: ";".join(safe_extract_labels(x))
        )
        
        # Garante que a primeira linha não seja vazia (o que o safe_extract_labels faz)
        # e escreve no CSV no modo 'a' (append) sem o cabeçalho e index
        labels_series.to_csv(
            OUTPUT_Y, 
            mode='a', 
            index=False, 
            header=False,
            encoding='utf-8'
        )
        
        current_index += n_rows
        
    except Exception as e:
        print(f"\n⚠️ ERRO (PULANDO ARQUIVO) ao processar {file_path.name}: {e}. A consolidação continua.")
        continue 

# Garante que todos os dados tenham sido escritos no disco
X_memmap.flush() 

print("\n----------------------------------------------------")
print("🚀 CONSOLIDAÇÃO DE TREINO/VAL CONCLUÍDA!")
print(f"Arquivos prontos: {OUTPUT_X} e {OUTPUT_Y}")
print("O próximo passo é rodar o 'optlstm.py' com o Gradient Clipping.")
print("----------------------------------------------------")