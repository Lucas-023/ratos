# terni.py — Versão Completa e Funcional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# Importa do seu dataloader.py
from dataloader import LazyFrameDataset, FEATURE_COLUMNS, TARGET_COLUMN

# =========================================================
# 1. Definição da Arquitetura do Modelo
# =========================================================

class MABeClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(MABeClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# =========================================================
# 2. Funções Auxiliares
# =========================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in tqdm(dataloader, desc="Treinando", leave=False):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        # Multi-label classification → BCEWithLogitsLoss
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
    
    return total_loss / len(dataloader.dataset)

# ====================================n=====================
# 3. Execução Principal
# =========================================================

if __name__ == "__main__":
    base_path = Path("MABe-mouse-behavior-detection/feature_engineered_data")
    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"Nenhum arquivo parquet encontrado em {base_path.resolve()}")

    print(f"✅ {len(parquet_files)} arquivos encontrados para treino.")

    dataset = LazyFrameDataset(parquet_files, FEATURE_COLUMNS, TARGET_COLUMN)
    
    # Divisão simples: 80% treino, 20% validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # =========================================================
    # 4. Inicialização do Modelo
    # =========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Treinando em: {device}")

    input_size = len(FEATURE_COLUMNS)
    num_classes = len(dataset.target_map)

    model = MABeClassifier(input_size=input_size, num_classes=num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # =========================================================
    # 5. Loop de Treinamento
    # =========================================================

    EPOCHS = 10
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        print(f"\n🌍 Época {epoch+1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"  🔹 Loss Treino: {train_loss:.6f} | 🔸 Loss Validação: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_mabe_model.pt")
            print("  💾 Novo melhor modelo salvo!")

    print("\n✅ Treinamento finalizado.")
    print(f"Melhor loss de validação: {best_val_loss:.6f}")
