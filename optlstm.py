import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Any
import os
import json
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# ============================================================
# 1. CONFIGURAÇÕES GERAIS E CAMINHOS
# ============================================================
# --- Caminhos dos Arquivos ---
CONSOLIDATED_X_PATH = Path("consolidated_X.npy")
CONSOLIDATED_Y_PATH = Path("consolidated_Y.csv")
BEHAVIOR_MAP_PATH = Path("behavior_map.json")
CLASS_WEIGHTS_PATH = Path("class_weights.npy") # <-- NOVO: Caminho para os pesos

# --- Hiperparâmetros de Treinamento ---
SEQ_LEN = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-4  # <-- AJUSTADO: Um ponto de partida melhor
WEIGHT_DECAY = 1e-5
N_EPOCHS = 50 # Aumente se necessário

# --- Configurações de Performance ---
# Use 'os.cpu_count()' para usar todos os cores, ou 0 no Windows se houver problemas
NUM_WORKERS = os.cpu_count() if os.name != 'nt' else 0
PIN_MEMORY = True

# --- Carregamento dos dados necessários ---
# Garante que os arquivos essenciais existem antes de começar
if not all([CONSOLIDATED_X_PATH.exists(), CONSOLIDATED_Y_PATH.exists(), BEHAVIOR_MAP_PATH.exists(), CLASS_WEIGHTS_PATH.exists()]):
    print("❌ ERRO: Um ou mais arquivos necessários não foram encontrados.")
    print("Certifique-se de que 'consolidated_X.npy', 'consolidated_Y.csv', 'behavior_map.json' e 'class_weights.npy' existem.")
    print("Rode o script 'calculate_weights.py' para gerar o mapa e os pesos.")
    exit()

# Carrega o mapa de comportamentos e os pesos de classe
with open(BEHAVIOR_MAP_PATH, 'r') as f:
    BEHAVIOR_MAP = json.load(f)
CLASS_WEIGHTS = torch.from_numpy(np.load(CLASS_WEIGHTS_PATH))
NUM_CLASSES = len(BEHAVIOR_MAP)
INPUT_SIZE = np.load(CONSOLIDATED_X_PATH, mmap_mode='r').shape[1]

print(f"✔️ Arquivos carregados. Input Size: {INPUT_SIZE}, Num Classes: {NUM_CLASSES}")


# ============================================================
# 2. IMPLEMENTAÇÃO DA FUNÇÃO DE PERDA (FOCAL LOSS COM PESOS)
# ============================================================
class BinaryFocalLoss(nn.Module):
    """Implementação Focal Loss para classificação Multi-Label com pesos de classe."""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight # <-- Armazena os pesos

    def forward(self, inputs, targets):
        # Transfere os pesos para o mesmo dispositivo que os inputs (GPU/CPU)
        pos_w = self.pos_weight.to(inputs.device) if self.pos_weight is not None else None

        # Calcula a Binary Cross-Entropy (BCE) Loss com os pesos para as classes positivas
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=pos_w)

        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# ============================================================
# 3. DATASET PERSONALIZADO (MEMMAP PARA EFICIÊNCIA)
# ============================================================
class MemmapSequenceDataset(Dataset):
    """Dataset que lê de arquivos .npy e .csv e gera sequências."""
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.X = np.load(CONSOLIDATED_X_PATH, mmap_mode='r')
        self.Y_df = pd.read_csv(CONSOLIDATED_Y_PATH, header=None, names=['behavior'])
        self.Y_df.dropna(inplace=True) # Remove linhas vazias

        self.num_classes = NUM_CLASSES
        self.behavior_map = BEHAVIOR_MAP
        
        # Garante que o número de labels corresponde ao de features
        num_features = self.X.shape[0]
        num_labels = len(self.Y_df)
        self.data_len = min(num_features, num_labels)

    def __len__(self):
        return self.data_len - self.seq_len + 1

    def __getitem__(self, index):
        # --- Features ---
        start_idx = index
        end_idx = index + self.seq_len
        sequence_x = self.X[start_idx:end_idx, :].astype(np.float32)

        # --- Label (do último frame da sequência) ---
        label_str = self.Y_df.iloc[end_idx - 1]['behavior']
        labels = label_str.split(';')
        
        # Multi-hot encoding
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if label in self.behavior_map:
                target[self.behavior_map[label]] = 1.0

        return torch.from_numpy(sequence_x), target


# ============================================================
# 4. MODELO LSTM (PYTORCH LIGHTNING)
# ============================================================
class LSTMBehaviorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate, lr, weight_decay, class_weights):
        super().__init__()
        self.save_hyperparameters() # Salva os hiperparâmetros

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # <-- MUDANÇA PRINCIPAL: Instancia a loss passando os pesos
        self.criterion = BinaryFocalLoss(gamma=2.0, pos_weight=class_weights)

    def forward(self, x):
        # LSTM retorna output, (hidden_state, cell_state)
        lstm_out, _ = self.lstm(x)
        # Pegamos apenas a saída do último passo da sequência
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Adiciona métricas de validação (opcional, mas recomendado)
        preds = torch.sigmoid(logits)
        # F1 Score é uma boa métrica para multi-label desbalanceado
        # from torchmetrics.functional import f1_score
        # f1 = f1_score(preds, y.int(), task='multilabel', num_labels=self.hparams.num_classes)
        # self.log('val_f1', f1, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # Scheduler para ajustar a taxa de aprendizado
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reduz quando a métrica monitorada para de diminuir
            factor=0.1,      # Fator de redução (new_lr = lr * factor)
            patience=3      # Número de épocas sem melhora antes de reduzir
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Métrica a ser monitorada
            },
        }

# ============================================================
# 5. EXECUÇÃO DO TREINAMENTO
# ============================================================
if __name__ == "__main__":
    pl.seed_everything(42) # Para reprodutibilidade

    print("Criando datasets e dataloaders...")
    full_dataset = MemmapSequenceDataset(seq_len=SEQ_LEN)
    
    # Divisão treino/validação
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    
    print(f"Dataset: {len(full_dataset)} amostras | Treino: {len(train_ds)} | Validação: {len(val_ds)}")

    # Instancia o modelo, passando os pesos de classe
    model = LSTMBehaviorModel(
        input_size=INPUT_SIZE,
        hidden_size=512,
        num_layers=5,
        num_classes=NUM_CLASSES,
        dropout_rate=0.4,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weights=CLASS_WEIGHTS # <-- Passa os pesos aqui!
    )

    # Callbacks para salvar o melhor modelo e parar cedo se não houver melhora
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=7, # Para o treino após 7 épocas sem melhora na val_loss
        verbose=True,
        mode='min'
    )

    # Configura o Trainer do PyTorch Lightning
    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    print("\n🚀 Iniciando o treinamento do modelo...")
    trainer.fit(model, train_loader, val_loader)
    print("✅ Treinamento concluído!")