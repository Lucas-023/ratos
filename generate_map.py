import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
from pathlib import Path

# =========================================================
# CONFIGURAÇÕES DE ARQUIVO
# =========================================================
CONSOLIDATED_Y_PATH = "consolidated_Y_FE.csv" 
BEHAVIOR_MAP_OUTPUT = "behavior_map.json" 

def generate_behavior_map():
    """
    Carrega o arquivo de labels, cria o mapeamento de classes (LabelEncoder) 
    e salva o resultado no formato JSON necessário para o teste de inferência.
    """
    
    print(f"Carregando labels de {CONSOLIDATED_Y_PATH}...")
    
    if not Path(CONSOLIDATED_Y_PATH).exists():
        print(f"❌ ERRO: Arquivo de labels '{CONSOLIDATED_Y_PATH}' não encontrado no diretório.")
        return

    # Carregar labels
    Y_df = pd.read_csv(CONSOLIDATED_Y_PATH)
    # Assume que o label está na primeira coluna
    Y_labels = Y_df.iloc[:, 0].astype(str).str.strip() 
    
    # Criar e ajustar o codificador
    le = LabelEncoder()
    le.fit(Y_labels)
    
    # O mapeamento: {índice numérico: nome do comportamento}
    behavior_map = {int(i): label for i, label in enumerate(le.classes_)}
    
    # Salvar o mapeamento em JSON
    with open(BEHAVIOR_MAP_OUTPUT, "w") as f:
        json.dump(behavior_map, f, indent=4)
        
    print(f"\n=======================================================")
    print(f"✅ Mapeamento de classes concluído.")
    print(f"O arquivo '{BEHAVIOR_MAP_OUTPUT}' foi criado com sucesso.")
    print(f"Total de classes encontradas: {len(le.classes_)}")
    print(f"Exemplo do mapa: {list(behavior_map.items())[:3]}...")
    print(f"=======================================================")

if __name__ == "__main__":
    generate_behavior_map()
