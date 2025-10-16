import numpy as np

print("Iniciando a verificação de 'consolidated_X.npy'...")

try:
    # Carrega o arquivo (agora salvo via np.save padrão, deve funcionar)
    X = np.load('consolidated_X.npy', mmap_mode='r', allow_pickle=True)
    
    if isinstance(X, np.ndarray):
        print(f"\n✅ Shape do array (Lido com sucesso): {X.shape}")
        
        # Amostra para checagem rápida
        X_sample = X[:100000] 
        has_nan = np.isnan(X_sample).any()
        has_inf = np.isinf(X_sample).any()
        
        print(f"Contém NaN (amostra): {has_nan}")
        print(f"Contém Inf (amostra): {has_inf}")
        
        X_max = np.max(X_sample)
        X_min = np.min(X_sample)
        print(f"Valor Máximo (Normalizado): {X_max:.2f}")
        print(f"Valor Mínimo (Normalizado): {X_min:.2f}")
            
        if X_max > 50 or X_min < -50 or has_nan or has_inf:
             print("\n❌ ERRO DE DADOS: A Normalização falhou ou ainda há corrupção. O treinamento VAI falhar.")
        else:
             print("\n🚀 DADOS PRONTOS! O arquivo está limpo e estável.")
             print("   Prossiga imediatamente para o optlstm.py.")

    else:
        print("❌ ERRO DE FORMATO: O arquivo não é um array NumPy. Falha na escrita.")

except Exception as e:
    print(f"\n❌ ERRO CRÍTICO: {e}")
    print("O arquivo 'consolidated_X.npy' continua corrompido. Tente instalar o NumPy novamente.")