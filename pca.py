import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
X_FILE = "consolidated_X_FE.npy" 
SAMPLE_SIZE = 50000 
VARIANCE_THRESHOLD = 0.95 
MASK_OUTPUT_FILE = "feature_mask_102.npy"

def load_and_scale_data(file_path: str, sample_size: int) -> tuple[np.ndarray, int]:
    """Carrega, amostra e padroniza os dados, removendo features de variância zero."""
    
    print(f"Carregando dados de {file_path}...")
    try:
        X_data_full = np.load(file_path, mmap_mode='r')
    except Exception as e:
        print(f"Erro ao carregar o arquivo {file_path}: {e}")
        return np.array([]), 0

    if X_data_full.shape[0] > sample_size:
        print(f"Amostrando {sample_size} frames para agilizar a PCA...")
        sample_indices = np.random.choice(X_data_full.shape[0], sample_size, replace=False)
        X_data_sample = X_data_full[sample_indices]
    else:
        print("Usando o dataset completo.")
        X_data_sample = X_data_full[:]
    
    original_features_count = X_data_sample.shape[1]
    print(f"Dimensões para PCA: {X_data_sample.shape}")
    
    # 2. Remoção de Features com Variância Quase Zero
    stds = np.std(X_data_sample, axis=0)
    non_zero_std_indices = stds > 1e-6 
    
    # 🚨 AÇÃO CRÍTICA: SALVANDO A MÁSCARA 🚨
    np.save(MASK_OUTPUT_FILE, non_zero_std_indices)
    print(f"✅ Máscara de features salva em {MASK_OUTPUT_FILE}.")
    
    X_clean = X_data_sample[:, non_zero_std_indices]
    
    print(f"Removidas {original_features_count - X_clean.shape[1]} features de variância zero.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    return X_scaled, original_features_count

def run_pca_analysis(scaled_data: np.ndarray, original_features: int):
    # ... (restante do código PCA, que funcionou anteriormente) ...
    if scaled_data.size == 0:
        print("Não há dados válidos para executar o PCA.")
        return

    print("\nExecutando PCA...")
    pca = PCA(n_components=None) 
    pca.fit(scaled_data)
    
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    n_components = np.where(cumulative_variance >= VARIANCE_THRESHOLD)[0][0] + 1
    
    print(f"=======================================================")
    print(f"✅ ANÁLISE PCA CONCLUÍDA")
    print(f"=======================================================")
    print(f"Número total de features ÚTEIS: {pca.n_components_}")
    print(f"Para capturar {VARIANCE_THRESHOLD*100}% da variância, são necessários:")
    print(f"➡️ {n_components} Componentes Principais")
    print(f"=======================================================")
    
    # Este código plota o gráfico, mas não vou incluí-lo para evitar poluição
    # plt.figure(figsize=(10, 6)); plt.plot(...); plt.show()


if __name__ == "__main__":
    scaled_data, original_features = load_and_scale_data(X_FILE, SAMPLE_SIZE)
    run_pca_analysis(scaled_data, original_features)