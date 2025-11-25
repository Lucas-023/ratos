import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Any, Iterable
import os
import pickle

# =========================================================
# CONFIGURAÇÕES E PARÂMETROS
# =========================================================

# Caminho para os arquivos Parquet processados para CatBoost
BASE_PATH_TRAIN = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost")

OUTPUT_X = "consolidated_X_catboost.npy"
OUTPUT_Y = "consolidated_Y_catboost.csv"
OUTPUT_CATEGORICAL_INFO = "categorical_info_catboost.pkl"  # Informações sobre variáveis categóricas

# =========================================================
# FUNÇÃO AUXILIAR: Extração Segura de Labels
# =========================================================

def safe_extract_labels(label_raw: Any) -> List[str]:
    """Trata formatos de labels e retorna uma lista de labels válidas."""
    if isinstance(label_raw, (list, np.ndarray, pd.Series)):
        return [str(l).strip() for l in label_raw if str(l).strip()]
    if pd.isna(label_raw):
        return []
    label_str = str(label_raw).strip()
    if not label_str or label_str.lower() in ('nan', '0.0', '0'):
        return []
    return [l.strip() for l in label_str.split(';') if l.strip()]


# =========================================================
# IDENTIFICAÇÃO DE COLUNAS
# =========================================================

def identify_feature_columns(df_sample: pd.DataFrame) -> tuple:
    """
    Identifica automaticamente as colunas de features, categóricas e numéricas.
    Retorna: (feature_cols, categorical_cols, metadata_cols)
    """
    # Colunas de metadados que não são features
    metadata_cols = [
        'frame', 'behavior', 'video_id', 'unique_frame_id',
        'lab_id', 'frames_per_second', 'video_duration_sec', 
        'pix_per_cm_approx', 'video_width_pix', 'video_height_pix',
        'body_parts_tracked', 'behaviors_labeled', 'tracking_method',
    ]
    
    # Identifica variáveis categóricas
    categorical_cols = []
    for col in df_sample.columns:
        if col in metadata_cols:
            continue
        if df_sample[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif df_sample[col].dtype == 'object':
            # Pode ser categórica se tiver poucos valores únicos
            n_unique = df_sample[col].nunique()
            if n_unique < 50 and n_unique < len(df_sample) * 0.1:
                categorical_cols.append(col)
    
    # Todas as outras colunas numéricas são features
    feature_cols = [
        col for col in df_sample.columns
        if col not in metadata_cols and col not in categorical_cols
        and df_sample[col].dtype in [np.float32, np.float64, np.int32, np.int64]
    ]
    
    return feature_cols, categorical_cols, metadata_cols


# =========================================================
# 1. PRÉ-CÁLCULO DO TAMANHO TOTAL
# =========================================================

print("🔍 Identificando estrutura dos dados...")

# -----------------------------------------------------------------
# Função utilitária para coletar arquivos Parquet (case-insensitive)
# -----------------------------------------------------------------
def collect_parquet_files(base_path: Path) -> List[Path]:
    patterns = ["*.parquet", "*.PARQUET", "*.Parquet"]
    files = []
    for pattern in patterns:
        files.extend(base_path.rglob(pattern))
    # Remove duplicados preservando ordem
    seen = set()
    unique_files = []
    for f in sorted(files):
        if f not in seen:
            unique_files.append(f)
            seen.add(f)
    return unique_files


# Carrega um arquivo de amostra para identificar colunas
parquet_files = collect_parquet_files(BASE_PATH_TRAIN)
if not parquet_files:
    print(f"❌ NENHUM arquivo Parquet encontrado em {BASE_PATH_TRAIN.resolve()}.")
    exit()

# Carrega amostra para identificar colunas
sample_file = parquet_files[0]
df_sample = pd.read_parquet(sample_file, engine='fastparquet')
FEATURE_COLS, CATEGORICAL_COLS, METADATA_COLS = identify_feature_columns(df_sample)

print(f"✅ Identificadas {len(FEATURE_COLS)} features numéricas")
print(f"✅ Identificadas {len(CATEGORICAL_COLS)} variáveis categóricas")
print(f"   Categóricas: {CATEGORICAL_COLS[:10]}..." if len(CATEGORICAL_COLS) > 10 else f"   Categóricas: {CATEGORICAL_COLS}")

TARGET_COLUMN = 'behavior'
N_FEATURES = len(FEATURE_COLS)

# Conta frames
total_frames = 0
print("\n🔍 Calculando o número total de frames...")
for file_path in tqdm(parquet_files, desc="Contando Frames"):
    try:
        df_temp = pd.read_parquet(file_path, engine='fastparquet', columns=[TARGET_COLUMN])
        total_frames += df_temp.shape[0]
    except Exception as e:
        print(f"\n⚠️ ERRO ao contar frames em {file_path.name}: {e}. Pulando.")

if total_frames == 0:
    print("❌ Nenhum frame válido encontrado para consolidação. Saindo.")
    exit()

print(f"✅ Total de frames a serem consolidados: {total_frames}")


# =========================================================
# 2. COLETA DE INFORMAÇÕES SOBRE VARIÁVEIS CATEGÓRICAS
# =========================================================

print("\n📊 Coletando informações sobre variáveis categóricas...")

categorical_info = {}
for col in CATEGORICAL_COLS:
    categorical_info[col] = {
        'categories': [],
        'dtype': 'category'
    }

for file_path in tqdm(parquet_files, desc="Analisando Categóricas"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet', columns=CATEGORICAL_COLS)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                unique_vals = df[col].dropna().unique()
                categorical_info[col]['categories'].extend([str(v) for v in unique_vals])
    except Exception as e:
        print(f"\n⚠️ ERRO ao analisar categóricas em {file_path.name}: {e}. Pulando.")

# Remove duplicatas e ordena
for col in CATEGORICAL_COLS:
    categorical_info[col]['categories'] = sorted(list(set(categorical_info[col]['categories'])))

# Salva informações sobre categóricas
with open(OUTPUT_CATEGORICAL_INFO, 'wb') as f:
    pickle.dump(categorical_info, f)

print(f"✅ Informações sobre categóricas salvas em {OUTPUT_CATEGORICAL_INFO}")


# =========================================================
# 3. CONSOLIDAÇÃO DE FEATURES NUMÉRICAS (SEM NORMALIZAÇÃO)
# =========================================================

print(f"\n💾 Consolidando features numéricas ({total_frames} frames x {N_FEATURES} features)...")

# Pré-aloca array (sem normalização - CatBoost não precisa)
X_memmap_temp = np.memmap(
    "temp_" + OUTPUT_X,
    dtype=np.float32,
    mode='w+',
    shape=(total_frames, N_FEATURES)
)

# Reseta o arquivo CSV de labels
with open(OUTPUT_Y, 'w', encoding='utf-8') as f_csv:
    f_csv.write(f"{TARGET_COLUMN}\n")

# Arquivo para variáveis categóricas (salvamos separadamente)
categorical_data = {col: [] for col in CATEGORICAL_COLS}

current_index = 0
print("🚀 Iniciando consolidação...")

# Estatísticas de diagnóstico
total_rows_processed = 0
total_labels_found = 0
files_with_labels = 0
files_without_labels = 0

for file_path in tqdm(parquet_files, desc="Consolidando Dados"):
    try:
        df = pd.read_parquet(file_path, engine='fastparquet')
        n_rows = len(df)
        
        # Verifica se a coluna de target existe
        if TARGET_COLUMN not in df.columns:
            tqdm.write(f"⚠️ Coluna '{TARGET_COLUMN}' não encontrada em {file_path.name}. Pulando labels.")
            # Ainda processa features, mas pula labels
            labels_series = pd.Series([pd.NA] * n_rows)
        else:
            # --- B) Labels (Y) ---
            labels_series = df[TARGET_COLUMN].apply(
                lambda x: ";".join(safe_extract_labels(x)) if safe_extract_labels(x) else pd.NA
            )
            # Converte strings vazias para NaN antes de salvar
            labels_series = labels_series.replace('', pd.NA)
            
            # Diagnóstico: conta labels encontrados neste arquivo
            labels_in_file = labels_series.notna().sum()
            if labels_in_file > 0:
                files_with_labels += 1
                total_labels_found += labels_in_file
                if files_with_labels <= 3:  # Mostra apenas os primeiros 3 arquivos com labels
                    tqdm.write(f"   ✅ {file_path.name}: {labels_in_file:,} labels encontrados de {n_rows:,} registros")
            else:
                files_without_labels += 1
        
        # --- A) Features Numéricas (X) ---
        # Seleciona apenas as colunas de features numéricas
        features_df = df.reindex(columns=FEATURE_COLS, fill_value=np.nan)
        features_np = features_df.values.astype(np.float32)
        
        # CatBoost lida bem com NaNs, mas podemos imputar com 0 para economizar espaço
        # Em produção, você pode deixar NaNs e o CatBoost tratará automaticamente
        features_np = np.nan_to_num(features_np, nan=0.0)
        
        # Escreve no memmap
        X_memmap_temp[current_index : current_index + n_rows] = features_np
        
        # Salva labels
        labels_series.to_csv(OUTPUT_Y, mode='a', index=False, header=False, encoding='utf-8', na_rep='')
        
        total_rows_processed += n_rows
        
        # --- C) Variáveis Categóricas ---
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                # Converte para string e trata NaNs
                cat_values = df[col].astype(str).fillna('nan').tolist()
                categorical_data[col].extend(cat_values)
            else:
                # Preenche com 'nan' se a coluna não existir
                categorical_data[col].extend(['nan'] * n_rows)
        
        current_index += n_rows
        
    except Exception as e:
        print(f"\n⚠️ ERRO ao consolidar {file_path.name}: {e}. Pulando.")
        import traceback
        traceback.print_exc()
        continue

# Garante que todos os dados foram escritos
X_memmap_temp.flush()

# Resumo de diagnóstico de labels
print("\n" + "="*60)
print("📊 RESUMO DE DIAGNÓSTICO DE LABELS")
print("="*60)
print(f"   • Total de registros processados: {total_rows_processed:,}")
print(f"   • Registros com labels: {total_labels_found:,}")
print(f"   • Registros sem labels: {total_rows_processed - total_labels_found:,}")
print(f"   • Arquivos com labels: {files_with_labels}")
print(f"   • Arquivos sem labels: {files_without_labels}")
if total_labels_found == 0:
    print("\n   ⚠️ ATENÇÃO: NENHUM label foi encontrado!")
    print("   Possíveis causas:")
    print("   1. A coluna 'behavior' não existe nos arquivos parquet")
    print("   2. Todos os valores de 'behavior' estão vazios/NaN")
    print("   3. O formato dos labels não está sendo reconhecido pela função safe_extract_labels")
    print("\n   💡 Verifique um arquivo parquet manualmente:")
    print(f"      df = pd.read_parquet('{parquet_files[0] if parquet_files else 'arquivo.parquet'}')")
    print("      print(df['behavior'].head(20))")
print("="*60)

# Salva array final
print("\n💾 Salvando array final de features...")
try:
    X_final_array = np.array(X_memmap_temp)
    np.save(OUTPUT_X, X_final_array)
    os.remove("temp_" + OUTPUT_X)
    print(f"✅ Features numéricas salvas em {OUTPUT_X}")
except Exception as e:
    print(f"\n❌ ERRO ao salvar features: {e}")
    print("O arquivo temporário pode estar disponível em temp_" + OUTPUT_X)

# Salva variáveis categóricas
print("\n💾 Salvando variáveis categóricas...")
categorical_df = pd.DataFrame(categorical_data)
categorical_output = OUTPUT_X.replace('.npy', '_categorical.parquet')
categorical_df.to_parquet(categorical_output, engine='fastparquet', index=False)
print(f"✅ Variáveis categóricas salvas em {categorical_output}")

print("\n" + "="*60)
print("✅ CONSOLIDAÇÃO PARA CATBOOST CONCLUÍDA")
print("="*60)
print(f"📁 Features numéricas: {OUTPUT_X}")
print(f"📁 Labels: {OUTPUT_Y}")
print(f"📁 Variáveis categóricas: {categorical_output}")
print(f"📁 Informações categóricas: {OUTPUT_CATEGORICAL_INFO}")
print("\n💡 Próximos passos:")
print("   1. Carregue os dados usando:")
print("      X = np.load('" + OUTPUT_X + "')")
print("      y = pd.read_csv('" + OUTPUT_Y + "')")
print("      cat_features = pd.read_parquet('" + categorical_output + "')")
print("   2. Use cat_features diretamente no CatBoost (sem OHE)")
print("   3. CatBoost lidará automaticamente com valores ausentes")


