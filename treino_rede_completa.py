# terni.py - Versão Final e Funcional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# IMPORTAÇÃO CRÍTICA: Importa a classe Dataset e as variáveis de colunas definidas
# no seu dataloader.py
from dataloader import LazyFrameDataset, FEATURE_COLUMNS, TARGET_COLUMN 

# =========================================================
# 1. Definição da Arquitetura do Modelo
# =========================================================

class MABeClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MABeClassifier, self).__init__()
        # 117 -> 256
        self.fc1 = nn.Linear(input_size, 256)
        # 256 -> 128
        self.fc2 = nn.Linear(256, 128)
        # 128 -> 37 (o número de classes de comportamento)
        self.fc3 = nn.Linear(128, num_classes)
        # Dropout para evitar Overfitting
        self.dropout = nn.Dropout(0.2)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x) 
        return x

# =========================================================
# 2. Configuração do Treinamento
# =========================================================

# Configuração de Hiperparâmetros
INPUT_SIZE = len(FEATURE_COLUMNS) # Tira o valor fixo e usa o valor importado
NUM_CLASSES = 37 # Será corrigido após a inicialização do Dataset
BATCH_SIZE = 64  
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10 

# Define o dispositivo: CUDA se disponível, caso contrário CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de treinamento: {DEVICE}")

# =========================================================
# 3. Inicialização e Loop de Treinamento
# =========================================================
if __name__ == "__main__":
    
    # 3a. Caminho para os arquivos Parquet processados (Ajuste se necessário)
    base_path = Path("MABe-mouse-behavior-detection/feature_engineered_data")
    parquet_files = list(base_path.rglob("*.parquet"))
    
    if not parquet_files:
        print("❌ Não foram encontrados arquivos Parquet. Verifique o caminho.")
    else:
        # 3b. Cria a instância do Dataset
        mouse_dataset = LazyFrameDataset(
            parquet_files=parquet_files,
            features=FEATURE_COLUMNS,
            target=TARGET_COLUMN
        )
        
        # O número de classes é extraído DO DATASET
        NUM_CLASSES = len(mouse_dataset.target_map)
        
        # Inicializa o modelo
        model = MABeClassifier(INPUT_SIZE, NUM_CLASSES).to(DEVICE)
        
        # Define a Função de Perda (Criterion) e o Otimizador
        criterion = nn.BCEWithLogitsLoss() 
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 3c. Cria o DataLoader
        train_loader = DataLoader(
            dataset=mouse_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,          
            num_workers=4,         
            pin_memory=True        
        )

        print(f"\n▶️ Iniciando Treinamento em {NUM_EPOCHS} épocas (Classes: {NUM_CLASSES}, Features: {INPUT_SIZE})...")

        # INÍCIO DO LOOP DE TREINAMENTO (AGORA DENTRO DO ESCOPO CORRETO)
        for epoch in range(NUM_EPOCHS):
            model.train() 
            total_loss = 0
            
            # Use tqdm para ter uma barra de progresso durante as épocas
            loop = tqdm(train_loader, desc=f"Época {epoch+1}/{NUM_EPOCHS}", leave=True)
            
            for X_batch, y_batch in loop:
                # Move os dados para o dispositivo (GPU/CPU)
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                # 1. Zero os gradientes acumulados
                optimizer.zero_grad()
                
                # 2. Forward Pass: calcular as saídas do modelo
                outputs = model(X_batch)
                
                # 3. Calcular a Perda
                loss = criterion(outputs, y_batch)
                
                # 4. Backward Pass: calcular os gradientes
                loss.backward()
                
                # 5. Otimizar: atualizar os pesos do modelo
                optimizer.step()
                
                total_loss += loss.item() * X_batch.size(0)
                
                # Atualiza a barra de progresso com a perda atual
                loop.set_postfix(loss=loss.item())

            # CORREÇÃO: Usa o tamanho TOTAL do dataset para calcular a perda média
            avg_loss = total_loss / len(mouse_dataset) 
            print(f"✅ Época {epoch+1} Concluída | Perda Média: {avg_loss:.4f}")
            
        print("\nTreinamento concluído!")
        
        # 4. Salvar o Modelo
        model_path = Path("mabe_classifier_final.pth")
        torch.save(model.state_dict(), model_path)
        print(f"✅ Modelo salvo em: {model_path}")