import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List, Any
from tqdm import tqdm
import os 
import json 
import torch.nn.functional as F # ✅ NOVO: Necessário para a BinaryFocalLoss

# ============================================================
# IMPLEMENTAÇÃO MANUAL DO FOCAL LOSS (CORREÇÃO DO ERRO)
# ============================================================
class BinaryFocalLoss(nn.Module):
    """Implementação Focal Loss para classificação Multi-Label (Binary Cross-Entropy)"""
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs são os logits (antes do sigmoid)
        
        # Binary Cross-Entropy (BCE) Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Ponderação do BCE (pt é a probabilidade do target, usada para focar a perda)
        pt = torch.exp(-BCE_loss)
        
        # Focal Loss: (1 - pt)^gamma * BCE
        F_loss = (1 - pt)**self.gamma * BCE_loss
        
        # Ponderação Alpha (opcional, mas incluída na implementação)
        if self.alpha is not None:
            # Alpha pondera as perdas de classes raras (targets=1) e majoritárias (targets=0)
            alpha_t = targets * self.alpha + (1. - targets) * (1. - self.alpha)
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# ============================================================
# CONFIGURAÇÕES DE PERFORMANCE MÁXIMA
# ============================================================
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium') 
    PIN_MEMORY = True 
else:
    PIN_MEMORY = False

# --- DEFINIÇÕES DE COLUNAS (MANTENDO A ESTRUTURA ORIGINAL) ---
try:
    from dataloader import FEATURE_COLUMNS, TARGET_COLUMN 
except ImportError:
    FEATURE_COLUMNS = [f'x_{i}' for i in range(1, 118)] 
    TARGET_COLUMN = 'behavior'

# ============================================================
# CONFIGURAÇÕES DO DATASET E TREINO
# ============================================================
CONSOLIDATED_X_PATH = "consolidated_X.npy"
CONSOLIDATED_Y_PATH = "consolidated_Y.csv"
X_MEAN_PATH = "X_mean.npy"
X_STD_PATH = "X_std.npy"
BEHAVIOR_MAP_OUTPUT = "behavior_map.json" 

SEQ_LEN = 30
INPUT_SIZE = len(FEATURE_COLUMNS) # 117
BATCH_SIZE = 256 
NUM_WORKERS = os.cpu_count() if os.name != 'nt' else 0 

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4    


# ============================================================
# 1. Dataset Otimizado com NumPy Memmap
# ============================================================

class MemmapSequenceDataset(Dataset):
    
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        
        print(f"Carregando labels (Y) de {CONSOLIDATED_Y_PATH}...")
        df_y = pd.read_csv(CONSOLIDATED_Y_PATH)
        self.Y_labels = df_y.iloc[:, 0].tolist() 
        
        self.total_frames = len(self.Y_labels) 
        self.n_features = INPUT_SIZE 
        
        print(f"Carregando features (X) de {CONSOLIDATED_X_PATH} via Memmap...")
        self.X_memmap = np.memmap(
            CONSOLIDATED_X_PATH, 
            dtype=np.float32, 
            mode='r', 
            shape=(self.total_frames, self.n_features) 
        )
        
        self.target_map = self._build_target_map()
        self.num_classes = len(self.target_map)
        
        self._save_behavior_map()
        
        self.valid_indices = np.arange(0, self.total_frames - self.seq_len)

        print(f"✅ Total de Sequências para Treino/Val: {len(self.valid_indices)}")
        print(f"✅ Total de Classes de Comportamento: {self.num_classes}")

    def safe_extract_labels(self, label_raw: Any) -> List[str]:
        if isinstance(label_raw, (list, np.ndarray, pd.Series)):
             return [str(l).strip() for l in label_raw if str(l).strip()]
        if pd.isna(label_raw): 
            return []
        label_str = str(label_raw).strip()
        if not label_str or label_str.lower() in ('nan', '0.0', '0'):
            return []
        return [l.strip() for l in label_str.split(';') if l.strip()]

    def _build_target_map(self):
        target_map = {}
        label_counter = 0
        
        for label_raw in tqdm(self.Y_labels, desc="Mapeando Labels", leave=False):
            labels_to_process = self.safe_extract_labels(label_raw)
            for label in labels_to_process:
                if label and label not in target_map:
                    target_map[label] = label_counter
                    label_counter += 1
        return target_map

    def _save_behavior_map(self):
        inverted_map = {v: k for k, v in self.target_map.items()}
        with open(BEHAVIOR_MAP_OUTPUT, "w") as f:
            json.dump(inverted_map, f, indent=4)
        print(f"✅ Mapa de Comportamento (índice -> nome) salvo em {BEHAVIOR_MAP_OUTPUT}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx] 
        
        X_np_seq = self.X_memmap[start_idx : start_idx + self.seq_len]
        X_seq = torch.from_numpy(X_np_seq.copy()) 
        
        Y_label_raw = self.Y_labels[start_idx + self.seq_len - 1]
        labels_present = self.safe_extract_labels(Y_label_raw)
        
        y_multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for label in labels_present:
            if label and label in self.target_map:
                y_multi_hot[self.target_map[label]] = 1.0 

        return X_seq, y_multi_hot


# ============================================================
# 2. Modelo LSTM (Com Focal Loss Manual)
# ============================================================

class LSTMBehaviorModel(pl.LightningModule):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout_rate, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        super().__init__()
        self.save_hyperparameters() 
        
        self.lstm = nn.LSTM(
            input_size, 
            self.hparams.hidden_size, 
            self.hparams.num_layers, 
            batch_first=True,
            dropout=self.hparams.dropout_rate if self.hparams.num_layers > 1 else 0 # Dropout só funciona entre camadas
        )
        self.fc = nn.Linear(self.hparams.hidden_size, self.hparams.num_classes)
        
        # ✅ NOVO: Usa a classe BinaryFocalLoss implementada localmente
        self.loss_fn = BinaryFocalLoss(gamma=0.5) 

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

    def training_step(self, batch, batch_idx):
        X_seq, y = batch
        logits = self(X_seq)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_seq, y = batch
        logits = self(X_seq)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # ✅ NOVO: Adiciona o Agendador (Scheduler)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',         # Monitora a redução da perda (minimização)
            factor=0.5,         # Reduz o LR pela metade
            patience=3        # Espera 3 épocas sem melhoria no val_loss
        )
        
        # O PyTorch Lightning espera um dicionário para schedulers
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', # Garante que o val_loss seja monitorado
            }
        }


# ============================================================
# 3. Script principal (Execução e Ponte)
# ============================================================

if __name__ == "__main__":

    if not all(Path(p).exists() for p in [CONSOLIDATED_X_PATH, CONSOLIDATED_Y_PATH, X_MEAN_PATH, X_STD_PATH]):
          print("----------------------------------------------------------------------")
          print("❌ ARQUIVOS CONSOLIDADOS OU ESTATÍSTICAS NÃO ENCONTRADOS.")
          print("----------------------------------------------------------------------")
          exit()
    
    full_dataset = MemmapSequenceDataset(seq_len=SEQ_LEN)
    
    total_sequences = len(full_dataset)
    train_size = int(0.8 * total_sequences)
    
    train_ds = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_ds = torch.utils.data.Subset(full_dataset, range(train_size, total_sequences))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = LSTMBehaviorModel(
        input_size=INPUT_SIZE,
        hidden_size=512, 
        num_layers=5, 
        dropout_rate = 0.4,
        num_classes=full_dataset.num_classes,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY

    )

    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm"
    )

    print("\n🚀 Iniciando treinamento com Focal Loss (Implementação Manual).")
    print("O Focal Loss deve resolver o viés de classe 'climb'.")
    trainer.fit(model, train_loader, val_loader)