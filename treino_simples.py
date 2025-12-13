import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import gc
from pathlib import Path

# =================================================================
# CONFIGURAÇÕES
# =================================================================
DATASET_PATH = Path("MABe-mouse-behavior-detection/ready_for_train/train_dataset_catboost.parquet")
MODEL_OUTPUT = "catboost_mouse_model_simple.cbm"

# Colunas a excluir
COLS_TO_DROP = ['frame', 'video_id', 'unique_frame_id', 'behavior', 'behaviors_labeled', 'lab_id']

print("🚀 Carregando dataset...")
df = pd.read_parquet(DATASET_PATH, engine='fastparquet')

# 1. Limpeza de Target
print("🧹 Limpando dados...")
mask_valid = (df['behavior'].notna()) & (df['behavior'] != 'None') & (df['behavior'] != 'nan')
df = df.loc[mask_valid].copy()
gc.collect()

y = df['behavior'].copy()
print(f"🧠 Classes: {y.nunique()}")
print(f"📊 Tamanho do dataset: {len(df):,} amostras")

# 2. Features
feature_cols = [c for c in df.columns if c not in COLS_TO_DROP]
X = df[feature_cols]

# 3. Categóricas
for col in X.columns:
    if X[col].dtype == 'object' or col in ['sex', 'strain', 'condition', 'mouse_id']:
        X[col] = X[col].astype(str).astype('category')

cat_features_indices = np.where((X.dtypes == 'category'))[0]

# 4. Split simples
print("✂️ Dividindo dados...")
test_size = int(0.2 * len(X))
test_idx = np.random.choice(len(X), test_size, replace=False)
train_mask = np.ones(len(X), dtype=bool)
train_mask[test_idx] = False

X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"📚 Treino: {len(X_train):,} | Teste: {len(X_test):,}")

# Libera memória
del df, X, y
gc.collect()

# 5. Cria Pools
train_pool = Pool(X_train, y_train, cat_features=cat_features_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_features_indices)

# 6. Treina modelo
print("\n🔥 Iniciando treinamento...")
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=5,
    auto_class_weights='SqrtBalanced',
    task_type="CPU",
    thread_count=-1,
    verbose=100,
    early_stopping_rounds=100,
    loss_function='MultiClass',
    eval_metric='TotalF1'
)

model.fit(train_pool, eval_set=test_pool, plot=False)

# 7. Salva modelo
print(f"\n💾 Salvando modelo em {MODEL_OUTPUT}...")
model.save_model(MODEL_OUTPUT)

# 8. Métricas
print("\n📊 Feature importance (Top 20):")
print(model.get_feature_importance(prettified=True).head(20))

print("\n✅ Treinamento concluído!")
