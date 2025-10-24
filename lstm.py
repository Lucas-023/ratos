import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import time

# =========================================================
# CONFIGURAÇÕES (AJUSTE AQUI)
# =========================================================
# Arquivos de dados
CONSOLIDATED_X_PATH = "consolidated_X_FE.npy" 
CONSOLIDATED_Y_PATH = "consolidated_Y_FE.csv" 
FEATURE_MASK_PATH = "feature_mask_102.npy"     

# Parâmetros do Modelo e Treinamento
SEQ_LEN = 10    
# INPUT_SIZE será determinado dinamicamente (102 ou 118)
HIDDEN_SIZE = 256 
NUM_LAYERS = 3    
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
N_EPOCHS = 20     
VAL_SIZE = 0.15   

# Configuração do dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# =========================================================
# 1. DEFINIÇÃO DA ESTRUTURA DO MODELO (LSTM OTIMIZADA)
# =========================================================

class OptimizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(OptimizedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM com 3 camadas e 256 hidden units, com dropout de 0.2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        
        # Camada de Classificação Final
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# =========================================================
# 2. DATASET CUSTOMIZADO (Para Lidar com o Memmap)
# =========================================================

class MemmapSequenceDataset(Dataset):
    def __init__(self, X_filtered, Y_encoded, seq_len):
        self.X = X_filtered
        self.Y = Y_encoded
        self.seq_len = seq_len
        self.num_sequences = len(X_filtered) - seq_len + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        x_sequence = self.X[idx : idx + self.seq_len, :]
        y_label = self.Y[idx + self.seq_len - 1]
        
        return torch.tensor(x_sequence, dtype=torch.float32), torch.tensor(y_label, dtype=torch.long)

# =========================================================
# 3. CARREGAMENTO E FILTRAGEM DOS DADOS (CRÍTICO)
# =========================================================

def load_and_filter_data(x_path, y_path, mask_path):
    """Carrega, filtra e prepara os dados de features e labels, retornando o input_size real."""
    
    # --- CARREGAMENTO DE FEATURES (X) ---
    print(f"Carregando {x_path}...")
    try:
        X_full = np.load(x_path, mmap_mode='r')
    except Exception as e:
        print(f"❌ ERRO FATAL ao carregar X: {e}")
        return None, None, 0
        
    # --- APLICAÇÃO DA MÁSCARA (FILTRAGEM) ---
    try:
        print(f"Carregando e aplicando máscara de features de {mask_path}...")
        feature_mask = np.load(mask_path)
        
        if feature_mask.shape[0] != X_full.shape[1]:
            raise ValueError(f"Dimensão da máscara ({feature_mask.shape[0]}) incompatível com X ({X_full.shape[1]}).")
            
        X_filtered = X_full[:, feature_mask] 
        real_input_size = X_filtered.shape[1]
        print(f"✅ Features filtradas: {X_full.shape[1]} -> {real_input_size} (Input Size real)")

    except FileNotFoundError:
        # AQUI ESTAVA O PROBLEMA DE 'global'. Agora apenas usamos o X_full
        real_input_size = X_full.shape[1]
        X_filtered = X_full
        print(f"❌ ERRO: Máscara {mask_path} não encontrada. Usando o dataset COMPLETO ({real_input_size} features).")
        print("⚠️ VOCÊ PODE ESTAR TREINANDO COM FEATURES DE VARIANCIA ZERO!")
    
    # --- CARREGAMENTO E ENCODING DE LABELS (Y) ---
    print(f"Carregando labels de {y_path}...")
    Y_df = pd.read_csv(y_path)
    Y_labels = Y_df.iloc[:, 0].astype(str).str.strip() 
    
    # Encontrar todas as classes únicas
    all_classes = Y_labels.unique()
    
    le = LabelEncoder()
    le.fit(all_classes)
    Y_encoded = le.transform(Y_labels)
    NUM_CLASSES = len(le.classes_)
    
    print(f"Classes encontradas: {NUM_CLASSES} | Exemplos: {le.classes_[:5]}...")
    
    return X_filtered, Y_encoded, NUM_CLASSES, real_input_size

# =========================================================
# 4. FUNÇÃO DE TREINAMENTO PRINCIPAL
# =========================================================

def train_model():
    start_time = time.time()
    
    # 1. Carregamento e Preparação de Dados (Captura o real_input_size)
    X_filtered, Y_encoded, NUM_CLASSES, real_input_size = load_and_filter_data(
        CONSOLIDATED_X_PATH, CONSOLIDATED_Y_PATH, FEATURE_MASK_PATH
    )
    
    if X_filtered is None: return

    # 2. Separação Treino/Validação
    split_index = int(len(X_filtered) * (1 - VAL_SIZE))
    X_train_data, X_val_data = X_filtered[:split_index], X_filtered[split_index:]
    Y_train_data, Y_val_data = Y_encoded[:split_index], Y_encoded[split_index:]

    # 3. Criação de Datasets e DataLoaders
    train_dataset = MemmapSequenceDataset(X_train_data, Y_train_data, SEQ_LEN)
    val_dataset = MemmapSequenceDataset(X_val_data, Y_val_data, SEQ_LEN)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 4. Inicialização do Modelo, Perda e Otimizador
    model = OptimizedLSTM(real_input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nIniciando Treinamento (LSTM {NUM_LAYERS}x{HIDDEN_SIZE}, Features={real_input_size})...")
    print(f"Tamanho do Treino: {len(train_dataset)} | Tamanho da Validação: {len(val_dataset)}")
    
    best_val_loss = float('inf')
    
    # 5. Loop de Treinamento
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # --- TREINAMENTO ---
        for inputs, targets in tqdm(train_loader, desc=f"Época {epoch+1}/{N_EPOCHS} (Treino)"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDAÇÃO ---
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Época {epoch+1}/{N_EPOCHS} (Validação)"):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_predictions / total_samples
        
        print(f" | Perda Treino: {avg_train_loss:.4f} | Perda Validação: {avg_val_loss:.4f} | Acerto Validação: {val_accuracy:.2f}%")
        
        # 6. Salvamento do Melhor Modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = f"best_optlstm_fe_3l_256h_seq{SEQ_LEN}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"➡️ Modelo salvo: Perda de Validação melhorou para {best_val_loss:.4f}")
            
    end_time = time.time()
    print(f"\nTreinamento concluído em {(end_time - start_time) / 60:.2f} minutos.")

if __name__ == "__main__":
    train_model()

