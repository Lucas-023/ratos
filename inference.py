import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
from pathlib import Path

# =========================================================
# 1. CONFIGURAÇÕES (DEVE SER IGUAL AO TREINAMENTO)
# =========================================================
# Arquivos
CONSOLIDATED_X_PATH = "consolidated_X_FE.npy" 
FEATURE_MASK_PATH = "feature_mask_102.npy"     
MODEL_PATH = "best_optlstm_fe_3l_256h_seq10.pth" # <-- Verifique se o nome do arquivo corresponde ao seu melhor modelo
BEHAVIOR_MAP_PATH = "behavior_map.json"
OUTPUT_CSV_PATH = "test_predictions_final.csv"

# Parâmetros do Modelo
SEQ_LEN = 10    
HIDDEN_SIZE = 256 
NUM_LAYERS = 3    
BATCH_SIZE = 2048 # Aumentamos o Batch Size para Inferência Rápida

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo para inferência: {DEVICE}")

# =========================================================
# 2. ESTRUTURA DO MODELO (DEVE SER IDÊNTICA AO TREINAMENTO)
# =========================================================

class OptimizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Estrutura idêntica à de treinamento
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2) 
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Apenas o último passo de tempo
        return out

# =========================================================
# 3. DATASET E LOADERS (Adaptado para Inferência)
# =========================================================

class InferenceSequenceDataset(Dataset):
    """Dataset para carregar todas as sequências de X, sem labels Y."""
    def __init__(self, X_filtered, seq_len):
        self.X = X_filtered
        self.seq_len = seq_len
        # Calculamos os índices que terão uma sequência completa
        self.valid_start_indices = np.arange(0, len(X_filtered) - self.seq_len + 1)
        self.num_sequences = len(self.valid_start_indices)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = self.valid_start_indices[idx]
        x_sequence = self.X[start_idx : start_idx + self.seq_len, :]
        
        # Retorna apenas a sequência (sem label)
        return torch.tensor(x_sequence, dtype=torch.float32)

# =========================================================
# 4. FUNÇÃO DE INFERÊNCIA PRINCIPAL
# =========================================================

def run_inference():
    
    # --- 4.1 Carregamento da Máscara e Dados ---
    print(f"Carregando features de {CONSOLIDATED_X_PATH}...")
    X_full = np.load(CONSOLIDATED_X_PATH, mmap_mode='r')
    
    # ⚠️ IMPORTANTE: Carregamos o X_full, mas o dataset só gera sequências
    # a partir do frame SEQ_LEN - 1. A coluna 'frame' será ajustada depois.
    
    if Path(FEATURE_MASK_PATH).exists():
        feature_mask = np.load(FEATURE_MASK_PATH)
        X_filtered = X_full[:, feature_mask] 
        actual_input_size = X_filtered.shape[1]
        print(f"✅ Features filtradas: {X_full.shape[1]} -> {actual_input_size} (102).")
    else:
        X_filtered = X_full
        actual_input_size = X_full.shape[1]
        print(f"❌ Máscara {FEATURE_MASK_PATH} não encontrada. Usando {actual_input_size} features.")

    # --- 4.2 Carregamento do Mapeamento de Comportamentos ---
    try:
        with open(BEHAVIOR_MAP_PATH, "r") as f:
            # O mapa salva {indice: comportamento}, e precisamos inverter isso
            behavior_map = {int(k): v for k, v in json.load(f).items()}
            num_classes = len(behavior_map)
            print(f"✅ Mapa de comportamentos carregado. Classes: {num_classes}")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de mapeamento {BEHAVIOR_MAP_PATH} não encontrado.")
        print("Você precisa rodar o script de treinamento primeiro para gerar o mapa de classes.")
        return

    # --- 4.3 Inicialização do Modelo e Carregamento de Pesos ---
    model = OptimizedLSTM(
        input_size=actual_input_size,
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        num_classes=num_classes
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Coloca o modelo em modo de avaliação (desativa dropout)
        print(f"✅ Pesos do modelo {MODEL_PATH} carregados com sucesso.")
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de modelo {MODEL_PATH} não encontrado.")
        print("Por favor, verifique se o modelo da Época 1 foi salvo corretamente.")
        return

    # --- 4.4 Preparação do DataLoader para Inferência ---
    inference_dataset = InferenceSequenceDataset(X_filtered, SEQ_LEN)
    inference_loader = DataLoader(inference_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Total de sequências para prever: {len(inference_dataset)}")

    # --- 4.5 Loop de Previsão ---
    all_predictions_indices = []
    
    print("\nIniciando Inferência (Previsão de Comportamentos)...")
    with torch.no_grad():
        for inputs in tqdm(inference_loader, desc="Processando Lotes"):
            inputs = inputs.to(DEVICE)
            
            outputs = model(inputs)
            # O torch.max retorna (valores_max, índices_max)
            _, predicted_indices = torch.max(outputs.data, 1)
            
            all_predictions_indices.extend(predicted_indices.cpu().numpy())

    # --- 4.6 Processamento dos Resultados e Salvamento ---
    
    # 1. Ajuste dos Frames: A previsão é para o frame final da sequência (SEQ_LEN - 1)
    # Se SEQ_LEN=10, a primeira previsão é para o Frame 9 (índice 9)
    # Os primeiros 9 frames não têm previsão, pois não há sequência de 10 antes deles.
    start_frame_index = SEQ_LEN - 1
    
    # Gerar a lista de frames a partir do primeiro frame com previsão
    predicted_frames = np.arange(start_frame_index, start_frame_index + len(all_predictions_indices))
    
    # 2. Tradução dos Índices para Comportamentos
    predicted_behaviors = [behavior_map.get(idx, "CLASSE_DESCONHECIDA") for idx in all_predictions_indices]

    # 3. Criação do DataFrame de Resultados
    results_df = pd.DataFrame({
        'frame': predicted_frames,
        'predicted_label_index': all_predictions_indices,
        'predicted_behavior': predicted_behaviors
    })
    
    # 4. Salvamento
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\n=======================================================")
    print(f"✅ INFERÊNCIA CONCLUÍDA")
    print(f"Resultados salvos em: {OUTPUT_CSV_PATH}")
    print(f"Frames Previstos (Início ao Fim): {predicted_frames[0]} a {predicted_frames[-1]}")
    print(f"Total de Previsões: {len(results_df)} frames.")
    print(f"=======================================================")

if __name__ == "__main__":
    run_inference()