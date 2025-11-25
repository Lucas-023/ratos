from __future__ import annotations

"""
Treinamento CatBoost com uso mínimo de RAM.

Fluxo:
1. Usa arrays memmap (on-disk) para não carregar tudo em RAM.
2. Faz o merge das features numéricas + categóricas em blocos.
3. Constrói a matriz multi-hot de labels também em memmap.
4. Treina esquema One-vs-Rest reutilizando os mesmos arquivos mapeados.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, hamming_loss

warnings.filterwarnings("ignore")

# =================================================================
# CONFIGURAÇÕES
# =================================================================

X_PATH = Path("consolidated_X_catboost.npy")
Y_PATH = Path("consolidated_Y_catboost.csv")
CATEGORICAL_PATH = Path("consolidated_X_catboost_categorical.parquet")
CATEGORICAL_INFO_PATH = Path("categorical_info_catboost.pkl")

# Caches on-disk (podem ser reutilizados entre execuções)
COMBINED_X_CACHE = Path("consolidated_X_catboost_with_cats.npy")
Y_MULTI_HOT_CACHE = Path("consolidated_Y_multi_hot.npy")

# Tamanho de bloco para processar dados sem estourar RAM
CHUNK_ROWS = 50_000


# =================================================================
# FUNÇÕES AUXILIARES
# =================================================================

def parse_multi_label(label_str: str) -> list[str]:
    """Converte string (com ';') em lista de labels."""
    if pd.isna(label_str) or label_str == "":
        return []
    return [lbl.strip() for lbl in str(label_str).split(";") if lbl.strip()]


def load_label_space(y_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Cria lista ordenada de labels e mapa label->indice."""
    all_labels = set()
    for label_str in y_df["behavior"]:
        all_labels.update(parse_multi_label(label_str))

    ordered_labels = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(ordered_labels)}
    return ordered_labels, label_to_idx


def ensure_multi_hot_cache(
    y_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    rebuild: bool = False,
) -> np.memmap:
    """Gera/abre o cache memmap das labels multi-hot."""
    n_samples = len(y_df)
    n_classes = len(label_to_idx)

    if Y_MULTI_HOT_CACHE.exists() and not rebuild:
        print(f"📂 Reutilizando cache de labels multi-hot: {Y_MULTI_HOT_CACHE}")
        return np.load(Y_MULTI_HOT_CACHE, mmap_mode="r+")

    print("🧱 Construindo cache multi-hot em disco...")
    y_multi_hot = np.lib.format.open_memmap(
        Y_MULTI_HOT_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_classes),
    )
    y_multi_hot[:] = 0.0

    for idx, label_str in enumerate(y_df["behavior"]):
        for label in parse_multi_label(label_str):
            if label in label_to_idx:
                y_multi_hot[idx, label_to_idx[label]] = 1.0

        if (idx + 1) % 100_000 == 0:
            print(f"   • Processados {idx+1:,} registros de labels...")

    y_multi_hot.flush()
    print(f"✅ Cache de labels salvo em {Y_MULTI_HOT_CACHE}")
    return y_multi_hot


def _build_cat_value_maps(categorical_info: dict) -> dict[str, dict[str, int]]:
    """Mapeia categoria -> índice inteiro para cada coluna."""
    value_maps: dict[str, dict[str, int]] = {}
    for col, info in categorical_info.items():
        categories = info.get("categories", [])
        value_maps[col] = {str(val): idx for idx, val in enumerate(categories)}
    return value_maps


def ensure_feature_cache(
    rebuild: bool,
    categorical_info: dict,
) -> tuple[np.memmap, list[int]]:
    """
    Gera/abre matriz de features (numéricas + categóricas codificadas) em memmap.
    O merge ocorre em blocos para mimetizar o comportamento do LazyFrameDataset.
    """
    X_numeric = np.load(X_PATH, mmap_mode="r")
    n_samples, n_numeric = X_numeric.shape

    parquet_file = pq.ParquetFile(CATEGORICAL_PATH)
    cat_columns = parquet_file.schema.names
    n_cat = len(cat_columns)
    total_features = n_numeric + n_cat

    cat_value_maps = _build_cat_value_maps(categorical_info)

    if COMBINED_X_CACHE.exists() and not rebuild:
        print(f"📂 Reutilizando cache de features: {COMBINED_X_CACHE}")
        combined = np.load(COMBINED_X_CACHE, mmap_mode="r+")
        cat_feature_indices = list(range(n_numeric, total_features))
        return combined, cat_feature_indices

    print("🧱 Construindo cache numérico + categórico em disco...")
    combined = np.lib.format.open_memmap(
        COMBINED_X_CACHE,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, total_features),
    )

    # Copia numéricas em blocos (evita cópia única gigantesca)
    for start in range(0, n_samples, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, n_samples)
        combined[start:end, :n_numeric] = X_numeric[start:end]
        if (start // CHUNK_ROWS) % 10 == 0:
            print(f"   • Copiando features numéricas: {end:,}/{n_samples:,}")

    # Codifica categóricas bloco a bloco
    row_start = 0
    for batch in parquet_file.iter_batches(batch_size=CHUNK_ROWS):
        batch_df = batch.to_pandas()
        batch_rows = len(batch_df)

        for col_idx, col in enumerate(cat_columns):
            mapping = cat_value_maps.get(col, {})
            unknown_idx = len(mapping)

            column_series = batch_df[col].astype("string").fillna("__unk__")
            encoded = column_series.map(
                lambda v: mapping.get(str(v), unknown_idx),
                na_action="ignore",
            ).astype(np.float32)
            combined[row_start : row_start + batch_rows, n_numeric + col_idx] = encoded.values

        row_start += batch_rows
        print(f"   • Codificando categóricas: {row_start:,}/{n_samples:,}")

    combined.flush()
    cat_feature_indices = list(range(n_numeric, total_features))
    print(f"✅ Cache completo salvo em {COMBINED_X_CACHE}")
    return combined, cat_feature_indices


def train_models(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cat_feature_indices: list[int],
    all_labels: list[str],
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.1,
) -> list[CatBoostClassifier | None]:
    """Treina um modelo CatBoost One-vs-Rest por classe."""
    n_classes = y_train.shape[1]
    models: list[CatBoostClassifier | None] = []

    print("\n🚀 Iniciando treinamento One-vs-Rest...")
    for class_idx in range(n_classes):
        if class_idx % 10 == 0:
            print(f"   → Classe {class_idx + 1}/{n_classes}")

        y_class = y_train[:, class_idx]
        if y_class.sum() == 0:
            models.append(None)
            continue

        model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            cat_features=cat_feature_indices,
            task_type="CPU",
        )

        model.fit(
            X_train,
            y_class,
            eval_set=(X_val, y_val[:, class_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        models.append(model)

    print(f"✅ {sum(m is not None for m in models)} modelos treinados")
    return models


def evaluate(models: list[CatBoostClassifier | None], X_val: np.ndarray, y_val: np.ndarray):
    """Calcula métricas multi-label."""
    n_classes = y_val.shape[1]
    y_pred_proba = np.zeros((X_val.shape[0], n_classes), dtype=np.float32)

    for class_idx, model in enumerate(models):
        if model is not None:
            y_pred_proba[:, class_idx] = model.predict_proba(X_val)[:, 1]

    y_pred = (y_pred_proba >= 0.5).astype(int)
    hamming = hamming_loss(y_val, y_pred)
    subset_accuracy = accuracy_score(y_val, y_pred)

    print("\n📊 Avaliação:")
    print(f"   • Hamming Loss: {hamming:.4f} (menor é melhor)")
    print(f"   • Subset Accuracy: {subset_accuracy:.4f} (maior é melhor)")


def save_models(models: list[CatBoostClassifier | None], all_labels: list[str], label_to_idx: dict[str, int], cat_feature_indices: list[int]):
    model_dir = Path("catboost_models")
    model_dir.mkdir(exist_ok=True)

    for class_idx, model in enumerate(models):
        if model is None:
            continue
        safe_label = all_labels[class_idx].replace(" ", "_")
        model_path = model_dir / f"catboost_class_{class_idx}_{safe_label}.cbm"
        model.save_model(str(model_path))

    label_info = {
        "all_labels": all_labels,
        "label_to_idx": label_to_idx,
        "n_classes": len(all_labels),
        "cat_feature_indices": cat_feature_indices,
    }
    with open(model_dir / "label_info.pkl", "wb") as f:
        pickle.dump(label_info, f)

    print(f"💾 Modelos e metadados salvos em {model_dir.resolve()}")


# =================================================================
# MAIN
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="Treinamento CatBoost com baixo uso de RAM.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Proporção para validação (contígua).")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recalcula caches de features/labels.")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    print("📂 Carregando labels...")
    y_df = pd.read_csv(Y_PATH)
    print(f"✅ Labels carregados: {len(y_df)} amostras")

    all_labels, label_to_idx = load_label_space(y_df)
    print(f"✅ {len(all_labels)} comportamentos únicos")

    with open(CATEGORICAL_INFO_PATH, "rb") as f:
        categorical_info = pickle.load(f)

    y_multi_hot = ensure_multi_hot_cache(y_df, label_to_idx, rebuild=args.rebuild_cache)
    X_combined, cat_feature_indices = ensure_feature_cache(args.rebuild_cache, categorical_info)

    total_samples = X_combined.shape[0]
    split_idx = int((1.0 - args.val_split) * total_samples)
    if split_idx == 0 or split_idx == total_samples:
        raise ValueError("Divisão treino/val inválida. Ajuste --val_split.")

    X_train = X_combined[:split_idx]
    X_val = X_combined[split_idx:]
    y_train = y_multi_hot[:split_idx]
    y_val = y_multi_hot[split_idx:]

    print(f"\n📊 Divisão dos dados (sem cópia):")
    print(f"   • Treino: {X_train.shape[0]} amostras")
    print(f"   • Validação: {X_val.shape[0]} amostras")

    models = train_models(
        X_train,
        X_val,
        y_train,
        y_val,
        cat_feature_indices,
        all_labels,
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
    )

    evaluate(models, X_val, y_val)
    save_models(models, all_labels, label_to_idx, cat_feature_indices)

    print("\n" + "=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO")
    print("=" * 60)
    print("\n💡 Dicas:")
    print("   • Ajuste --iterations/--depth/--learning-rate conforme memória disponível.")
    print("   • Rode com --rebuild-cache se mudar os arquivos consolidados.")
    print("   • Para validação mais aleatória, gere splits no consolidate antes do cache.")


if __name__ == "__main__":
    main()
