import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from pathlib import Path

# =========================================================
# CONFIGURAÇÕES DE CAMINHO
# =========================================================

# Ajuste o caminho conforme necessário, mas geralmente é este:
CONSOLIDATED_X_PATH = Path("consolidated_X.npy")

# =========================================================
# 1. Carregamento e Pré-processamento dos Dados
# =========================================================

def load_and_scale_data(file_path: Path):
    """Carrega os dados e aplica a padronização (StandardScaler)."""
    if not file_path.exists():
        print(f"❌ ERRO: Arquivo {file_path} não encontrado.")
        print("Certifique-se de que 'consolidate_data.py' foi executado.")
        return None
    
    print(f"Carregando dados de {file_path.name}...")
    # Usamos o np.memmap para carregar, mas para PCA, o NumPy precisa carregar
    # o array completo (ou uma amostra), pois a PCA requer a matriz completa.
    X_data = np.load(file_path)
    
    # Se o dataset for grande (> 1 milhão de frames), amostre para acelerar a PCA!
    # Exemplo: Amostrar 50.000 frames aleatoriamente
    if X_data.shape[0] > 1000000:
        sample_size = 50000
        print(f"Amostrando {sample_size} frames para agilizar a PCA...")
        indices = np.random.choice(X_data.shape[0], sample_size, replace=False)
        X_data_sample = X_data[indices]
    else:
        X_data_sample = X_data
        
    print(f"Dimensões para PCA: {X_data_sample.shape}")
    
    # Padronização: A PCA é sensível à escala. É crucial padronizar os dados.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data_sample)
    
    return X_scaled, X_data_sample.shape[1] # Retorna dados padronizados e o número original de features

# =========================================================
# 2. Análise PCA e Visualização
# =========================================================

def run_pca_analysis(X_scaled: np.ndarray, n_features_original: int):
    """Executa a PCA e gera o gráfico de variância explicada."""
    print("\nExecutando PCA...")
    
    # Criamos o PCA sem especificar n_components para capturar toda a variância
    pca = PCA()
    pca.fit(X_scaled)
    
    # Variância Explicada Acumulada
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # -----------------------------------------------------------
    # Encontrando o número de componentes para 90% e 95% da variância
    # -----------------------------------------------------------
    
    # Para 90%
    n_components_90 = np.where(cumulative_variance >= 0.90)[0]
    n_90 = n_components_90[0] + 1 if len(n_components_90) > 0 else n_features_original
    
    # Para 95%
    n_components_95 = np.where(cumulative_variance >= 0.95)[0]
    n_95 = n_components_95[0] + 1 if len(n_components_95) > 0 else n_features_original

    # -----------------------------------------------------------
    # Geração do Gráfico (Scree Plot)
    # -----------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Curva da Variância Acumulada
    plt.plot(cumulative_variance, marker='o', linestyle='-', color='b')
    
    # Linhas de referência para 90% e 95%
    plt.axvline(n_90 - 1, color='r', linestyle='--', label=f'90% Variância ({n_90} Comp.)')
    plt.axvline(n_95 - 1, color='g', linestyle='--', label=f'95% Variância ({n_95} Comp.)')
    plt.axhline(0.90, color='r', linestyle=':')
    plt.axhline(0.95, color='g', linestyle=':')
    
    plt.title('PCA: Variância Explicada Acumulada')
    plt.xlabel('Número de Componentes Principais')
    plt.ylabel('Variância Explicada Acumulada')
    plt.grid(True)
    plt.legend()
    plt.xlim([0, min(n_features_original, 60)]) # Limita o eixo X para melhor visualização
    plt.show()

    print("\n=======================================================")
    print(f"Total de Features Originais: {n_features_original}")
    print("-------------------------------------------------------")
    
    if n_90 < n_features_original * 0.5:
        print(f"⚠️ Forte Indício de Redundância/Correlação.")
    
    print(f"✅ 90% da Variância é explicada por: {n_90} componentes.")
    print(f"✅ 95% da Variância é explicada por: {n_95} componentes.")
    print("=======================================================")
    
    return n_95 # Retorna o número de componentes para 95%

# =========================================================
# 3. Execução Principal
# =========================================================

if __name__ == "__main__":
    
    # Certifique-se de que as bibliotecas estão instaladas
    try:
        import sklearn
        import matplotlib
    except ImportError:
        print("❌ ERRO: Bibliotecas scikit-learn ou matplotlib não encontradas.")
        print("Execute: pip install scikit-learn matplotlib")
        exit()
        
    scaled_data, original_features = load_and_scale_data(CONSOLIDATED_X_PATH)
    
    if scaled_data is not None:
        run_pca_analysis(scaled_data, original_features)