import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from pathlib import Path
import gc

# =================================================================
# 1. CONFIGURAÇÕES
# =================================================================
DATASET_PATH = Path("MABe-mouse-behavior-detection/ready_for_train/train_dataset_catboost.parquet")
MODEL_OUTPUT = "catboost_mouse_model.cbm"

# Definimos explicitamente quais colunas NÃO são features de treino
METADATA_COLS = [
    'frame', 'video_id', 'unique_frame_id', 'lab_id', 
    'behavior', 'behaviors_labeled' 
]

def load_and_prepare_data():
    print(f"🚀 Carregando dataset otimizado: {DATASET_PATH}...")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado! Rode o consolidate_data_catboost.py primeiro.")

    # Carrega o dataset completo
    df = pd.read_parquet(DATASET_PATH)
    print(f"✅ Dados carregados: {df.shape[0]:,} linhas, {df.shape[1]} colunas")

    # =================================================================
    # 2. LIMPEZA E SEPARAÇÃO DE FEATURES (X) E TARGET (y)
    # =================================================================
    
    # Target inicial
    y = df['behavior']
    
    # 2.1 Remove linhas nulas
    mask_valid = y.notna() & (y != 'None') & (y != 'nan')
    df = df[mask_valid]
    y = y[mask_valid]

    # 2.2 FILTRO DE CLASSES RARAS (CORREÇÃO DO ERRO)
    # Conta quantas vezes cada comportamento aparece
    class_counts = y.value_counts()
    
    # Identifica classes com menos de 5 exemplos (número seguro para Cross-Validação)
    rare_classes = class_counts[class_counts < 5].index
    
    if len(rare_classes) > 0:
        print(f"\n⚠️ REMOVENDO CLASSES RARAS ( < 5 exemplos):")
        for cls in rare_classes:
            print(f"   - '{cls}': {class_counts[cls]} ocorrência(s)")
        
        # Filtra o dataset removendo essas classes
        mask_common = ~y.isin(rare_classes)
        df = df[mask_common]
        y = y[mask_common]
        print(f"✅ Classes raras removidas. Novo total: {len(df):,} linhas.")
    
    # Identifica features (Tudo que não é metadado nem target)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols]

    # Identifica índices das colunas categóricas automaticamente
    cat_features_indices = np.where(X.dtypes == 'category')[0]
    cat_features_names = X.columns[cat_features_indices].tolist()
    
    print(f"\n🔍 Features selecionadas ({len(feature_cols)}):")
    print(f"   {feature_cols}")
    print(f"🔍 Categóricas detectadas ({len(cat_features_names)}):")
    print(f"   {cat_features_names}")

    # Limpa memória do DF original
    del df
    gc.collect()

    # =================================================================
    # 3. DIVISÃO TREINO / VALIDAÇÃO
    # =================================================================
    print("\n✂️ Dividindo em Treino (80%) e Validação (20%)...")
    
    # Agora o stratify vai funcionar porque garantimos minímo de 5 amostras
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Treino: {X_train.shape[0]:,} amostras")
    print(f"   Validação: {X_val.shape[0]:,} amostras")

    # =================================================================
    # 4. CRIAÇÃO DOS POOLS DO CATBOOST
    # =================================================================
    print("\n🏊 Criando Pools do CatBoost...")
    
    train_pool = Pool(
        data=X_train, 
        label=y_train, 
        cat_features=cat_features_indices
    )
    
    val_pool = Pool(
        data=X_val, 
        label=y_val, 
        cat_features=cat_features_indices
    )

    return train_pool, val_pool

def train_model(train_pool, val_pool):
    print("\n🔥 Configurando e Iniciando Treinamento...")
    
    # Detecta GPU automaticamente, se falhar usa CPU
    task_type = "GPU"
    try:
        import torch
        if not torch.cuda.is_available():
            task_type = "CPU"
            print("⚠️ GPU não detectada (Torch), usando CPU.")
    except:
        pass # Tenta GPU mesmo assim, o CatBoost vai reclamar se não der

    print(f"   Modo de Treino: {task_type}")

    model = CatBoostClassifier(
        iterations=2000,           
        learning_rate=0.1,         
        depth=6,                   
        loss_function='MultiClass', 
        eval_metric='Accuracy',    
        task_type=task_type,      # 'GPU' ou 'CPU'
        devices='0',               
        verbose=100,               
        early_stopping_rounds=100  
    )

    try:
        model.fit(
            train_pool,
            eval_set=val_pool,
            plot=False
        )
    except Exception as e:
        if "GPU" in str(e):
            print("\n⚠️ Falha na GPU detectada. Reiniciando treino em CPU...")
            model.set_params(task_type="CPU", devices=None)
            model.fit(train_pool, eval_set=val_pool, plot=False)
        else:
            raise e
    
    print(f"\n💾 Salvando modelo em {MODEL_OUTPUT}...")
    model.save_model(MODEL_OUTPUT)
    print("✅ Modelo salvo com sucesso!")
    
    return model

if __name__ == "__main__":
    try:
        train_pool, val_pool = load_and_prepare_data()
        model = train_model(train_pool, val_pool)
        
        print("\n📊 Importância das Features (Top 10):")
        importance = model.get_feature_importance(prettified=True)
        print(importance.head(10))
        
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")