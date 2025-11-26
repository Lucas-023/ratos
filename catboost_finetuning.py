import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from pathlib import Path
import gc
import json

# =================================================================
# CONFIGURAÇÕES
# =================================================================
DATASET_PATH = Path("MABe-mouse-behavior-detection/ready_for_train/train_dataset_catboost.parquet")
MODEL_OUTPUT = "catboost_mouse_model_tuned.cbm"

METADATA_COLS = [
    'frame', 'video_id', 'unique_frame_id', 'lab_id', 
    'behavior', 'behaviors_labeled' 
]

def load_and_prepare_data():
    print(f"🚀 Carregando dataset: {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    # Target e Limpeza
    y = df['behavior']
    mask_valid = y.notna() & (y != 'None') & (y != 'nan')
    df = df[mask_valid]
    y = y[mask_valid]

    # Filtro de Classes Raras (< 50 exemplos para estabilidade)
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 50].index
    if len(rare_classes) > 0:
        print(f"⚠️ Removendo classes muito raras: {list(rare_classes)}")
        mask_common = ~y.isin(rare_classes)
        df = df[mask_common]
        y = y[mask_common]

    # Separação X / y
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols]

    # Detecção de Categóricas
    cat_features_indices = np.where((X.dtypes == 'category') | (X.dtypes == 'object'))[0]
    
    # Divisão
    print(f"✂️ Dividindo {len(X)} amostras...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Limpeza de memória
    del df, X, y
    gc.collect()

    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)
    
    return train_pool, val_pool

def train_tuned_model(train_pool, val_pool):
    print("\n🔥 Iniciando Treinamento TUNADO (Versão Corrigida GPU)...")
    
    model = CatBoostClassifier(
        # --- PARÂMETROS DE POTÊNCIA ---
        iterations=1000,            
        learning_rate=0.05,         # Baixo para aprender devagar e constante
        depth=8,                    # Profundo para capturar interações complexas
        
        auto_class_weights='SqrtBalanced',
        # --- ESTABILIDADE & REGULARIZAÇÃO ---
        l2_leaf_reg=5,              # Regularização L2 (Segura a onda do overfitting)
        random_strength=1,          # Substituto do RSM para GPU (Adiciona aleatoriedade segura)
        border_count=128,           # Otimização para GPU (Padrão é 128, ajuda na velocidade)
        
        # --- CONFIGURAÇÃO ---
        loss_function='MultiClass', 
        eval_metric='TotalF1',    
        task_type="GPU",           
        devices='0',               
        verbose=100,
        early_stopping_rounds=200,   
        
        # Otimização de memória GPU
        gpu_ram_part=0.9            
    )

    model.fit(train_pool, eval_set=val_pool, plot=False)
    
    print(f"💾 Salvando: {MODEL_OUTPUT}")
    model.save_model(MODEL_OUTPUT)
    
    return model

if __name__ == "__main__":
    try:
        train_pool, val_pool = load_and_prepare_data()
        model = train_tuned_model(train_pool, val_pool)
        
        print("\n📊 Importância das Features (Top 20):")
        print(model.get_feature_importance(prettified=True).head(20))
        
    except Exception as e:
        print(f"❌ Erro: {e}")