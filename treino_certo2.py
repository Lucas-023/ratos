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
    
    # 1. Limpeza de Target
    y = df['behavior']
    mask_valid = y.notna() & (y != 'None') & (y != 'nan')
    df = df[mask_valid].reset_index(drop=True)
    y = y[mask_valid].reset_index(drop=True)

    print(f"🧠 Total de classes (Target): {y.nunique()}")

    # 2. DEFINIÇÃO DE FEATURES
    feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]
    X = df[feature_cols]
    
    groups = df['video_id']
    
    # --- TRATAMENTO DE CATEGÓRICAS ---
    print("\n⚙️ Processando tipos de dados...")
    for col in X.columns:
        # Detecta colunas de texto (possíveis metadados)
        if X[col].dtype == 'object' or col in ['sex', 'strain', 'condition', 'mouse_id']:
            X[col] = X[col].astype(str).astype('category')

    cat_features_indices = np.where((X.dtypes == 'category'))[0]
    
    # ==========================================================================
    # 🕵️‍♂️ RELATÓRIO DE VARIÁVEIS (O CHECK QUE VOCÊ PEDIU)
    # ==========================================================================
    print("\n" + "="*60)
    print(f"🧐 INSPEÇÃO FINAL DE VARIÁVEIS ({len(X.columns)} features)")
    print("="*60)
    
    # 1. Verifica Metadados Específicos
    potential_meta = ['sex', 'strain', 'age', 'condition', 'pix_per_cm_approx', 'frames_per_second']
    print("📍 Status dos Metadados:")
    for meta in potential_meta:
        if meta in X.columns:
            print(f"   ✅ {meta:<20} (Tipo: {X[meta].dtype})")
        else:
            print(f"   ❌ {meta:<20} (Ausente/Removido)")
            
    # 2. Verifica se lab_id saiu mesmo
    if 'lab_id' not in X.columns:
        print("   ✅ lab_id              (Removido com sucesso para generalização)")
    else:
        print("   ⚠️ PERIGO: lab_id ainda está presente!")

    # 3. Lista Categóricas Detectadas
    cat_cols = X.select_dtypes(include=['category']).columns.tolist()
    print(f"\n🏷️  Features Categóricas Detectadas ({len(cat_cols)}):")
    print(f"   {cat_cols}")

    # 4. Lista Completa (Compacta)
    print(f"\n📋 Lista Completa de Todas as {len(X.columns)} Variáveis:")
    all_cols = sorted(list(X.columns))
    # Imprime em linhas de 4 em 4 para ficar legível
    for i in range(0, len(all_cols), 4):
        print("   " + " | ".join([f"{c:<25}" for c in all_cols[i:i+4]]))
    
    print("="*60 + "\n")
    
    # Pausa dramática para você ler (opcional, pode tirar o input se for rodar automático)
    # input("Pressione ENTER para confirmar e iniciar o treino...") 
    
    # ==========================================================================

    # 3. DIVISÃO POR VÍDEO
    print(f"✂️ Dividindo por grupos de vídeo ({groups.nunique()} vídeos)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    groups_val = groups.iloc[val_idx]

    # 4. CIRURGIA DE VÍDEO COMPLETO
    train_classes = set(y_train.unique())
    val_classes = set(y_val.unique())
    missing_in_train = val_classes - train_classes
    
    if missing_in_train:
        print(f"⚠️ DETECTADO: {len(missing_in_train)} classes raras só na validação.")
        print("🔧 Movendo VÍDEOS INTEIROS para o treino...")
        
        mask_rare = y_val.isin(missing_in_train)
        videos_to_rescue = groups_val[mask_rare].unique()
        mask_move = groups_val.isin(videos_to_rescue)
        
        X_train = pd.concat([X_train, X_val[mask_move]])
        y_train = pd.concat([y_train, y_val[mask_move]])
        
        X_val = X_val[~mask_move]
        y_val = y_val[~mask_move]
        print(f"✅ Resgatados {len(videos_to_rescue)} vídeos.")

    print(f"📚 Treino: {len(X_train)} | ✅ Validação: {len(X_val)}")
    
    del df, X, y, groups, groups_val
    gc.collect()

    train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
    val_pool = Pool(X_val, y_val, cat_features=cat_features_indices)
    
    return train_pool, val_pool

def train_tuned_model(train_pool, val_pool):
    print("\n🔥 Iniciando Treinamento com METADADOS (Sem Lab_ID)...")
    
    model = CatBoostClassifier(
        iterations=5000,            
        learning_rate=0.08,         
        depth=6,                    
        auto_class_weights='SqrtBalanced',
        
        border_count=32,           
        gpu_cat_features_storage='CpuPinnedMemory', 
        max_ctr_complexity=1,      
        
        l2_leaf_reg=5,              
        random_strength=1,          
        loss_function='MultiClass', 
        eval_metric='TotalF1',      
        task_type="GPU",           
        devices='0',               
        verbose=100,
        early_stopping_rounds=300,  
        gpu_ram_part=0.90            
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