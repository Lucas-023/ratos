import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

# =================================================================
# CONFIGURAÇÕES
# =================================================================
INPUT_DIR = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost")
OUTPUT_DIR = Path("MABe-mouse-behavior-detection/ready_for_train")
OUTPUT_FILE = OUTPUT_DIR / "train_dataset_catboost.parquet"

# Colunas que devem ser tratadas como categóricas (para economizar memória e ajudar o CatBoost)
CATEGORICAL_COLS = [
    'arena_type', 'arena_shape',
    'mouse1_sex', 'mouse2_sex', 'mouse3_sex', 'mouse4_sex',
    'mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain',
    'mouse1_color', 'mouse2_color', 'mouse3_color', 'mouse4_color',
    'mouse1_condition', 'mouse2_condition', 'mouse3_condition', 'mouse4_condition',
    'lab_id', 'tracking_method', 'behavior'
]

def optimize_dtypes(df):
    """Otimiza os tipos de dados para reduzir uso de memória."""
    for col in df.columns:
        if col in CATEGORICAL_COLS:
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def consolidate_data():
    print("🚀 Iniciando consolidação dos dados para CatBoost...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_files = list(INPUT_DIR.rglob("*.parquet"))
    print(f"🔍 Total de arquivos processados encontrados: {len(all_files)}")
    
    data_frames = []
    files_with_labels = 0
    total_rows = 0
    
    # Barra de progresso
    for file_path in tqdm(all_files, desc="Lendo e filtrando arquivos"):
        try:
            # Lê apenas as colunas necessárias para verificar se tem label
            # (Lê o arquivo todo pois parquet colunar é rápido)
            df = pd.read_parquet(file_path)
            
            # Filtra apenas linhas que têm comportamento anotado (não nulo)
            # Para treino supervisionado, frames sem label não ajudam diretamente
            if 'behavior' in df.columns:
                df_labeled = df[df['behavior'].notna() & (df['behavior'] != 'nan') & (df['behavior'] != 'None')].copy()
                
                if not df_labeled.empty:
                    # Otimiza memória antes de adicionar à lista
                    df_labeled = optimize_dtypes(df_labeled)
                    data_frames.append(df_labeled)
                    files_with_labels += 1
                    total_rows += len(df_labeled)
            
            # Limpeza de memória
            del df
            
        except Exception as e:
            print(f"⚠️ Erro ao ler {file_path.name}: {e}")
            continue

    if not data_frames:
        print("❌ Nenhum dado com label encontrado para consolidar!")
        return

    print(f"\n🧩 Concatenando {files_with_labels} arquivos ({total_rows:,} linhas)...")
    full_df = pd.concat(data_frames, ignore_index=True)
    
    # Libera memória da lista
    del data_frames
    gc.collect()
    
    print("💾 Salvando arquivo final consolidado...")
    print(f"   Destino: {OUTPUT_FILE}")
    
    # Salva em Parquet com compressão snappy (rápido e eficiente)
    full_df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    
    print("\n" + "="*50)
    print("✅ CONSOLIDAÇÃO FINALIZADA COM SUCESSO")
    print("="*50)
    print(f"📊 Dataset Final: {full_df.shape[0]:,} linhas e {full_df.shape[1]} colunas")
    print(f"🧠 Memória estimada do DF: {full_df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    # Verifica distribuição das classes
    if 'behavior' in full_df.columns:
        print("\n📈 Distribuição de Classes (Top 10):")
        print(full_df['behavior'].value_counts().head(10))

if __name__ == "__main__":
    consolidate_data()