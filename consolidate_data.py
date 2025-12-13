import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
import os

# =================================================================
# CONFIGURAÇÕES
# =================================================================
# Aponta para a pasta V4 onde os arquivos estão aparecendo agora
INPUT_DIR = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost_v4")
OUTPUT_DIR = Path("MABe-mouse-behavior-detection/ready_for_train")
OUTPUT_FILE = OUTPUT_DIR / "train_dataset_catboost.parquet"

# Lista completa de categóricas (incluindo as novas do train.csv)
CATEGORICAL_COLS = [
    # Gerais
    'lab_id', 'behavior', 'arena_type', 'arena_shape', 'tracking_method',
    
    # Ratos (Sexo, Linhagem, Cor, Condição, ID)
    'mouse1_sex', 'mouse1_strain', 'mouse1_color', 'mouse1_condition', 'mouse1_id',
    'mouse2_sex', 'mouse2_strain', 'mouse2_color', 'mouse2_condition', 'mouse2_id',
    'mouse3_sex', 'mouse3_strain', 'mouse3_color', 'mouse3_condition', 'mouse3_id',
    'mouse4_sex', 'mouse4_strain', 'mouse4_color', 'mouse4_condition', 'mouse4_id'
]

def optimize_dtypes(df):
    """Converte tipos para economizar memória RAM."""
    for col in df.columns:
        if col in CATEGORICAL_COLS:
            df[col] = df[col].astype(str).astype('category')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def consolidate_data():
    print("🚀 Iniciando consolidação FINAL...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # LIMPEZA PREVENTIVA: Apaga o arquivo antigo para não duplicar dados
    if OUTPUT_FILE.exists():
        print(f"🗑️ Removendo arquivo antigo: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    if not INPUT_DIR.exists():
        print(f"❌ ERRO: A pasta {INPUT_DIR} não existe.")
        return

    all_files = list(INPUT_DIR.rglob("*.parquet"))
    print(f"🔍 Encontrados {len(all_files)} arquivos na pasta v4.")
    
    if len(all_files) == 0:
        print("❌ A pasta v4 está vazia! Espere o processamento terminar.")
        return
    
    data_frames = []
    total_rows = 0
    
    for file_path in tqdm(all_files, desc="Lendo arquivos"):
        try:
            df = pd.read_parquet(file_path)
            
            # Filtra linhas com label válido (Supervisionado)
            if 'behavior' in df.columns:
                mask = df['behavior'].notna() & (df['behavior'] != 'None') & (df['behavior'] != 'nan')
                df = df[mask]
                
                if not df.empty:
                    df = optimize_dtypes(df)
                    data_frames.append(df)
                    total_rows += len(df)
            
            del df
            
        except Exception as e:
            print(f"⚠️ Erro ao ler {file_path.name}: {e}")
            continue

    if not data_frames:
        print("❌ Nenhum dado válido encontrado!")
        return

    print(f"\n🧩 Concatenando {len(data_frames)} arquivos ({total_rows:,} linhas)...")
    full_df = pd.concat(data_frames, ignore_index=True)
    
    del data_frames
    gc.collect()
    
    print(f"💾 Salvando em: {OUTPUT_FILE}")
    full_df.to_parquet(OUTPUT_FILE, index=False, compression='snappy')
    
    print("\n" + "="*50)
    print("✅ CONSOLIDAÇÃO CONCLUÍDA")
    print("="*50)
    print(f"📊 Colunas Finais: {list(full_df.columns)}")
    
    # Validação
    if 'mouse1_color' in full_df.columns:
        print("✅ Sucesso: Metadados novos (cor, arena) estão presentes!")
    else:
        print("⚠️ Aviso: Metadados novos não encontrados.")

if __name__ == "__main__":
    consolidate_data()