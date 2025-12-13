import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
import gc
import json
import sys

# =================================================================
# CONFIGURAÇÕES
# =================================================================
DATASET_PATH = Path("MABe-mouse-behavior-detection/ready_for_train/train_dataset_catboost.parquet")
MODEL_OUTPUT = "catboost_mouse_model_meta_v2.cbm" 

# --- LISTA DE EXCLUSÃO ---
# O que estiver aqui NÃO ENTRA no treino.
COLS_TO_DROP = [
    'frame', 
    'video_id',          
    'unique_frame_id', 
    'behavior',          # Target
    'behaviors_labeled',
    'lab_id'             # REMOVIDO para evitar vício em laboratório específico
]

def load_and_prepare_data():
    print(f"🚀 Carregando dataset: {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    # 1. Limpeza de Target - otimizado para grandes datasets
    print("🧹 Limpando dados inválidos...")
    mask_valid = (df['behavior'].notna()) & (df['behavior'] != 'None') & (df['behavior'] != 'nan')
    df = df.loc[mask_valid].copy()  # Use .loc[] in-place filtering before copy
    gc.collect()  # Force garbage collection
    
    y = df['behavior'].copy()

    print(f"🧠 Total de classes (Target): {y.nunique()}")

    # 2. DEFINIÇÃO DE FEATURES
    feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]
    X = df[feature_cols]
    
    groups = df['video_id'].copy()
    
    # --- TRATAMENTO DE CATEGÓRICAS ---
    print("\n⚙️ Processando tipos de dados...")
    for col in X.columns:
        # Detecta colunas de texto (possíveis metadados)
        if X[col].dtype == 'object' or col in ['sex', 'strain', 'condition', 'mouse_id']:
            X[col] = X[col].astype(str).astype('category')

    cat_features_indices = np.where((X.dtypes == 'category'))[0]
    
    print("\n✅ Dataset preparado. Iniciando processamento...")
    print("="*60 + "\n")

    # 3. DIVISÃO POR VÍDEO - SIMPLIFIED FOR SPEED
    print(f"✂️ Dividindo dados (estratégia simplificada)...")
    # Para dataset grande, usa simples random split sem stratification
    from sklearn.model_selection import train_test_split
    
    try:
        # Tenta com stratificação
        X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
            X, y, groups, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Se falhar (classes raras), faz split simples
        print("   ⚠️ Algumas classes têm < 2 amostras, usando split simples...")
        X_train, X_val, y_train, y_val, groups_train, groups_val = train_test_split(
            X, y, groups, test_size=0.2, random_state=42
        )
    
    print(f"✂️ Split realizado...")

    # 4. ENSURE NO CLASS ONLY IN VALIDATION
    train_classes = set(y_train.unique())
    val_classes = set(y_val.unique())
    missing_in_train = val_classes - train_classes
    
    # Keep moving rare classes until all validation classes are in training
    iterations = 0
    while missing_in_train and iterations < 10:
        iterations += 1
        print(f"⚠️ Iteração {iterations}: {len(missing_in_train)} classes só na validação")
        
        # Move ALL samples of rare classes to training
        mask_rare = y_val.isin(missing_in_train)
        n_moved = mask_rare.sum()
        
        X_train = pd.concat([X_train, X_val[mask_rare]], ignore_index=True)
        y_train = pd.concat([y_train, y_val[mask_rare]], ignore_index=True)
        
        # Remove from validation
        X_val = X_val[~mask_rare].reset_index(drop=True)
        y_val = y_val[~mask_rare].reset_index(drop=True)
        
        print(f"   ✅ Moved {n_moved} samples to training")
        
        # Update class sets for next iteration
        train_classes = set(y_train.unique())
        val_classes = set(y_val.unique())
        missing_in_train = val_classes - train_classes
    
    # Final verification
    if missing_in_train:
        print(f"❌ ERROR: Still have {len(missing_in_train)} classes in validation not in training!")
        print(f"   Classes: {missing_in_train}")
        sys.exit(1)

    print(f"✅ All validation classes are in training!")
    print(f"📚 Treino: {len(X_train):,} | ✅ Validação: {len(X_val):,}")
    
    # --- SAMPLING TO AVOID OUT OF MEMORY ---
    # Keep max 1.5M training samples but maintain class distribution
    max_train_samples = 1500000
    if len(X_train) > max_train_samples:
        print(f"\n⚠️ Reducing training set from {len(X_train):,} to {max_train_samples:,} (memory management)...")
        
        # Sample uniformly across all classes to maintain distribution
        indices = []
        unique_classes = y_train.unique()
        samples_per_class = max_train_samples // len(unique_classes)
        
        for class_label in unique_classes:
            class_mask = (y_train == class_label).values
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) > samples_per_class:
                sampled = np.random.choice(class_indices, samples_per_class, replace=False)
            else:
                sampled = class_indices
            indices.extend(sampled)
        
        indices = np.array(indices[:max_train_samples])
        X_train = X_train.iloc[indices]
        y_train = y_train.iloc[indices]
        
        # Re-verify classes after sampling
        if not (set(y_train.unique()) == set(y_val.unique())):
            print("⚠️ Some validation classes lost in sampling, moving them back...")
            missing_after_sample = set(y_val.unique()) - set(y_train.unique())
            if missing_after_sample:
                mask_rare = y_val.isin(missing_after_sample)
                X_train = pd.concat([X_train, X_val[mask_rare]], ignore_index=True)
                y_train = pd.concat([y_train, y_val[mask_rare]], ignore_index=True)
                X_val = X_val[~mask_rare].reset_index(drop=True)
                y_val = y_val[~mask_rare].reset_index(drop=True)
        
        print(f"✅ New training size: {len(X_train):,}")
    
    del df, X, y, groups, groups_train, groups_val
    gc.collect()

    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)
    
    return train_pool, val_pool

def train_tuned_model(train_pool, val_pool):
    print("\n🔥 Iniciando Treinamento com METADADOS (Sem Lab_ID)...")
    
    model = CatBoostClassifier(
        iterations=2000,            
        learning_rate=0.08,         
        depth=6,                    
        auto_class_weights='SqrtBalanced',
        
        border_count=32,           
        max_ctr_complexity=1,      
        
        l2_leaf_reg=5,              
        random_strength=1,          
        loss_function='MultiClass', 
        eval_metric='TotalF1',      
        task_type="GPU",           
        thread_count=-1,           
        verbose=100,
        early_stopping_rounds=300  
    )

    model.fit(train_pool, eval_set=val_pool, plot=False)
    
    print(f"💾 Salvando: {MODEL_OUTPUT}")
    model.save_model(MODEL_OUTPUT)
    
    import json
    with open("classes_list.json", "w") as f:
        json.dump(list(model.classes_), f)
        
    return model

if __name__ == "__main__":
    try:
        train_pool, val_pool = load_and_prepare_data()
        model = train_tuned_model(train_pool, val_pool)
        
        print("\n📊 Importância das Features (Top 25):")
        print(model.get_feature_importance(prettified=True).head(25))
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()