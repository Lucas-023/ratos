"""
Exemplo de treinamento com CatBoost usando os dados processados.

Este script demonstra como:
1. Carregar os dados consolidados
2. Preparar variáveis categóricas
3. Treinar um modelo CatBoost
4. Fazer predições multi-label
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# CONFIGURAÇÕES
# =================================================================

X_PATH = "consolidated_X_catboost.npy"
Y_PATH = "consolidated_Y_catboost.csv"
CATEGORICAL_PATH = "consolidated_X_catboost_categorical.parquet"
CATEGORICAL_INFO_PATH = "categorical_info_catboost.pkl"

# =================================================================
# CARREGAMENTO DE DADOS
# =================================================================

print("📂 Carregando dados...")

# Carrega features numéricas
X = np.load(X_PATH)
print(f"✅ Features numéricas carregadas: {X.shape}")

# Carrega labels
y_df = pd.read_csv(Y_PATH)
print(f"✅ Labels carregados: {len(y_df)} amostras")

# Processa labels para multi-label
def parse_multi_label(label_str):
    """Converte string de labels separados por ';' em lista."""
    if pd.isna(label_str) or label_str == '':
        return []
    return [l.strip() for l in str(label_str).split(';') if l.strip()]

# Cria matriz multi-label
all_labels = set()
for label_str in y_df['behavior']:
    labels = parse_multi_label(label_str)
    all_labels.update(labels)

all_labels = sorted(list(all_labels))
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
n_classes = len(all_labels)

print(f"✅ Total de classes de comportamento: {n_classes}")
print(f"   Classes: {all_labels[:10]}..." if len(all_labels) > 10 else f"   Classes: {all_labels}")

# Cria matriz multi-hot
y_multi_hot = np.zeros((len(y_df), n_classes), dtype=np.float32)
for idx, label_str in enumerate(y_df['behavior']):
    labels = parse_multi_label(label_str)
    for label in labels:
        if label in label_to_idx:
            y_multi_hot[idx, label_to_idx[label]] = 1.0

print(f"✅ Matriz multi-label criada: {y_multi_hot.shape}")

# Carrega variáveis categóricas
cat_features_df = pd.read_parquet(CATEGORICAL_PATH)
print(f"✅ Variáveis categóricas carregadas: {cat_features_df.shape}")

# Carrega informações sobre categóricas
with open(CATEGORICAL_INFO_PATH, 'rb') as f:
    categorical_info = pickle.load(f)

print(f"✅ Informações sobre categóricas carregadas")

# =================================================================
# PREPARAÇÃO DE VARIÁVEIS CATEGÓRICAS
# =================================================================

# Identifica índices das colunas categóricas no array X
# Como as categóricas estão em um DataFrame separado, precisamos concatená-las
# ou usar apenas as numéricas. Para este exemplo, vamos usar apenas as numéricas
# e adicionar as categóricas como features adicionais.

# Converte categóricas para índices numéricos (CatBoost requer isso)
cat_features_encoded = {}
cat_feature_indices = []

# Adiciona categóricas como colunas adicionais ao X
# (Alternativamente, você pode usar cat_features como parâmetro separado no CatBoost)
for col_idx, col in enumerate(cat_features_df.columns):
    # Converte para índices categóricos
    unique_vals = categorical_info[col]['categories']
    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
    
    # Mapeia valores para índices
    encoded = cat_features_df[col].apply(
        lambda x: val_to_idx.get(str(x), len(unique_vals))  # Usa último índice para valores não vistos
    ).values
    
    # Adiciona como coluna numérica ao X
    X = np.column_stack([X, encoded.astype(np.float32)])
    cat_feature_indices.append(X.shape[1] - 1)  # Índice da última coluna adicionada

print(f"✅ Variáveis categóricas adicionadas. Total de features: {X.shape[1]}")
print(f"   Índices das categóricas: {cat_feature_indices}")

# =================================================================
# DIVISÃO TREINO/VALIDAÇÃO
# =================================================================

# Para multi-label, podemos usar train_test_split normalmente
X_train, X_val, y_train, y_val = train_test_split(
    X, y_multi_hot,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print(f"\n📊 Divisão dos dados:")
print(f"   Treino: {X_train.shape[0]} amostras")
print(f"   Validação: {X_val.shape[0]} amostras")

# =================================================================
# TREINAMENTO COM CATBOOST
# =================================================================

print("\n🚀 Treinando modelo CatBoost...")

# Para multi-label, treinamos um classificador por classe (One-vs-Rest)
# ou usamos CatBoost com loss='MultiLogloss' (se suportado)

# Opção 1: One-vs-Rest (mais comum para multi-label)
models = []
for class_idx in range(n_classes):
    if class_idx % 10 == 0:
        print(f"   Treinando classe {class_idx+1}/{n_classes}...")
    
    y_class = y_train[:, class_idx]
    
    # Pula classes sem exemplos positivos
    if y_class.sum() == 0:
        models.append(None)
        continue
    
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False,
        cat_features=cat_feature_indices,  # Especifica quais colunas são categóricas
        task_type='CPU',  # Mude para 'GPU' se disponível
    )
    
    # Treina
    model.fit(
        X_train, y_class,
        eval_set=(X_val, y_val[:, class_idx]),
        early_stopping_rounds=50,
        verbose=False
    )
    
    models.append(model)

print(f"✅ {sum(1 for m in models if m is not None)} modelos treinados")

# =================================================================
# AVALIAÇÃO
# =================================================================

print("\n📊 Avaliando modelo...")

# Predições
y_pred_proba = np.zeros((X_val.shape[0], n_classes))
for class_idx, model in enumerate(models):
    if model is not None:
        y_pred_proba[:, class_idx] = model.predict_proba(X_val)[:, 1]

# Converte probabilidades em predições binárias (threshold=0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)

# Métricas
hamming = hamming_loss(y_val, y_pred)
subset_accuracy = accuracy_score(y_val, y_pred)

print(f"   Hamming Loss: {hamming:.4f} (menor é melhor)")
print(f"   Subset Accuracy: {subset_accuracy:.4f} (maior é melhor)")

# =================================================================
# SALVAMENTO DO MODELO
# =================================================================

print("\n💾 Salvando modelos...")

model_dir = Path("catboost_models")
model_dir.mkdir(exist_ok=True)

# Salva cada modelo
for class_idx, model in enumerate(models):
    if model is not None:
        model_path = model_dir / f"catboost_class_{class_idx}_{all_labels[class_idx]}.cbm"
        model.save_model(str(model_path))

# Salva informações sobre labels
label_info = {
    'all_labels': all_labels,
    'label_to_idx': label_to_idx,
    'n_classes': n_classes,
    'cat_feature_indices': cat_feature_indices
}

with open(model_dir / "label_info.pkl", 'wb') as f:
    pickle.dump(label_info, f)

print(f"✅ Modelos salvos em {model_dir}")

print("\n" + "="*60)
print("✅ TREINAMENTO CONCLUÍDO")
print("="*60)
print("\n💡 Dicas para melhorar o modelo:")
print("   1. Ajuste hiperparâmetros (depth, learning_rate, iterations)")
print("   2. Use GPU se disponível (task_type='GPU')")
print("   3. Experimente diferentes thresholds para predições binárias")
print("   4. Use validação cruzada para avaliação mais robusta")
print("   5. Considere usar CatBoost com MultiLogloss se disponível")

