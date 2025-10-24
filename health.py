import numpy as np
from pathlib import Path
from typing import Tuple

# Ajuste o caminho para o seu novo arquivo consolidado
CONSOLIDATED_X_PATH = "consolidated_X_FE.npy" 
SAMPLE_SIZE = 100000  # Amostra maior para garantir a representatividade

def check_data_health(file_path: Path, sample_size: int) -> Tuple[float, float, float]:
    """
    Carrega o dataset e verifica a porcentagem de zeros, NaNs e outliers.
    """
    if not file_path.exists():
        print(f"❌ Arquivo não encontrado: {file_path}")
        return 0.0, 0.0, 0.0

    print(f"🔍 Carregando dados de {file_path} para verificação...")
    X = np.load(file_path)
    
    # Garante que temos dados suficientes para amostrar
    if X.shape[0] < sample_size:
        sample_size = X.shape[0]

    # Amostra aleatória para eficiência
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[sample_indices]
    
    total_elements = X_sample.size
    
    # 1. Zeros
    num_zeros = np.sum(X_sample == 0.0)
    percent_zeros = (num_zeros / total_elements) * 100

    # 2. NaNs (deve ser 0 após a consolidação)
    num_nans = np.sum(np.isnan(X_sample))
    percent_nans = (num_nans / total_elements) * 100

    # 3. Outliers (Valores Absolutos Altos - Esperado para dados Z-score)
    # Valores > 3 ou < -3 são considerados outliers na distribuição normal
    num_outliers = np.sum(np.abs(X_sample) > 3.0)
    percent_outliers = (num_outliers / total_elements) * 100

    return percent_zeros, percent_nans, percent_outliers

if __name__ == "__main__":
    
    # --- Execute o process_data.py e o consolidate_data.py ANTES de rodar este script ---
    
    zeros, nans, outliers = check_data_health(Path(CONSOLIDATED_X_PATH), SAMPLE_SIZE)

    print("\n=======================================================")
    print(f"✅ VERIFICAÇÃO DE SAÚDE DOS DADOS ({Path(CONSOLIDATED_X_PATH).name})")
    print("=======================================================")
    print(f"Total de Amostras verificadas: {SAMPLE_SIZE} frames.")
    print(f"Porcentagem de ZEROS (Imputação de NaN/Ratos Ausentes): {zeros:.4f}%")
    print(f"Porcentagem de NaN (Erro de Processamento): {nans:.4f}%")
    print(f"Porcentagem de Outliers (>3 desvios padrão): {outliers:.4f}%")
    
    if zeros > 10.0:
        print("\n⚠️ AVISO: A alta porcentagem de zeros sugere que muitos dados ainda estão faltando ou que muitos ratos estão ausentes.")
    if nans > 0.0001:
        print("\n❌ ERRO: NaNs foram encontrados no arquivo final. A imputação falhou durante a consolidação.")
    if nans < 0.0001 and zeros < 10.0:
        print("\n✅ SAÚDE DOS DADOS CONFIRMADA. O dataset está pronto para o treinamento.")

    print("=======================================================")