# train_lstm_optimized.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple
from tqdm import tqdm
from typing import Tuple

# Importa as definições de colunas do seu dataloader original
from dataloader import FEATURE_COLUMNS, TARGET_COLUMN

# ============================================================
# CONFIGURAÇÕES DO DATASET CONSOLIDADO
# ============================================================
CONSOLIDATED_X_PATH = "consolidated_X.npy"
CONSOLIDATED_Y_PATH = "consolidated_Y.csv"
SEQ_LEN = 30
INPUT_SIZE = len(FEATURE_COLUMNS) # 117
BATCH_SIZE = 256 # Mantemos o batch size alto para saturar a GPU
NUM_WORKERS = 16 # Mantemos o num_workers alto para alimentar a GPU

# ============================================================
# 1. Dataset Otimizado com NumPy Memmap
# ============================================================

class MemmapSequenceDataset(Dataset):
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        
        # 1. Carrega Labels (Y) para determinar o tamanho total
        print(f"Carregando labels (Y) de {CONSOLIDATED_Y_PATH}...")
        df_y = pd.read_csv(CONSOLIDATED_Y_PATH)
        # Pega a coluna de strings de labels, assume que é a primeira coluna
        self.Y_labels = df_y.iloc[:, 0].tolist() 
        
        self.total_frames = len(self.Y_labels) # <--- TAMANHO CRÍTICO
        self.n_features = INPUT_SIZE          # INPUT_SIZE deve ser 117
        
        # 2. Mapeamento de Features (X) - USANDO np.memmap DIRETAMENTE
        print(f"Carregando features (X) de {CONSOLIDATED_X_PATH} via Memmap...")
        self.X_memmap = np.memmap(
            CONSOLIDATED_X_PATH, 
            dtype=np.float32, 
            mode='r', # Modo 'r' para leitura
            # Força o mapeamento do arquivo binário cru com o tamanho calculado
            shape=(self.total_frames, self.n_features) 
        )
        
        # 3. Mapeamento de Comportamentos
        self.target_map = self._build_target_map()
        self.num_classes = len(self.target_map)
        
        # Índices válidos (apenas aqueles que cabem na sequência)
        self.valid_indices = np.arange(0, self.total_frames - self.seq_len)

        print(f"✅ Total de Frames: {self.total_frames}")
        print(f"✅ Total de Sequências: {len(self.valid_indices)}")
        print(f"✅ Total de Classes de Comportamento: {self.num_classes}")



    def _build_target_map(self):
        """Reconstrói o mapa de targets para Multi-Hot Encoding."""
        target_map = {}
        label_counter = 0
        
        for label_raw in tqdm(self.Y_labels, desc="Mapeando Labels", leave=False):
            # CORREÇÃO: Converte para string e lida com NaN (que é lido como float)
            label_str = str(label_raw)
            if label_str in ('nan', '', '0.0'): # Ignora vazios e NaN
                continue
            
            # A string está separada por ';'
            labels_to_process = label_str.split(";")
            
            for label in labels_to_process:
                if label and label not in target_map:
                    target_map[label] = label_counter
                    label_counter += 1
        return target_map

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pega o índice real no memmap
        # start_idx é o índice do frame inicial da sequência no array X gigante.
        start_idx = self.valid_indices[idx] 
        
        # 1. Leitura Ultra-Rápida das Features (X)
        # Pega o bloco contíguo de seq_len frames do arquivo consolidado via memmap
        X_np_seq = self.X_memmap[start_idx : start_idx + self.seq_len]
        
        # 2. Converte o array NumPy para tensor PyTorch
        # X_seq terá o shape (SEQ_LEN, INPUT_SIZE) -> (30, 117)
        X_seq = torch.from_numpy(X_np_seq)
        
        # 3. Processamento do Target (último frame da sequência)
        
        # Pega o label RAW (pode ser a string 'comportamento;outro' ou 'nan')
        Y_label_raw = self.Y_labels[start_idx + self.seq_len - 1]
        
        # CORREÇÃO: Converte para string e lida com NaN (lido como float)
        Y_label_str = str(Y_label_raw)
        
        if Y_label_str in ('nan', ''):
             # Se for vazio ou NaN, a lista de labels está vazia
             labels_present = []
        else:
             # Caso contrário, divide a string pelos delimitadores (;)
             labels_present = Y_label_str.split(";")
        
        # 4. Multi-Hot Encoding do Target (y)
        # Cria um vetor de zeros do tamanho do número de classes
        y_multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for label in labels_present:
            # Garante que o label não seja vazio e exista no nosso mapa
            if label and label in self.target_map:
                y_multi_hot[self.target_map[label]] = 1.0 

        # 5. Retorna o par Feature Sequence (X) e Label (y)
        return X_seq, y_multi_hot


# ============================================================
# 2. Modelo LSTM (mantido da sua versão)
# ============================================================

class LSTMBehaviorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

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
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================
# 3. Script principal (Execução)
# ============================================================

if __name__ == "__main__":

    # Verifica se os arquivos consolidados existem antes de prosseguir
    if not Path(CONSOLIDATED_X_PATH).exists() or not Path(CONSOLIDATED_Y_PATH).exists():
         print("----------------------------------------------------------------------")
         print("❌ ARQUIVOS CONSOLIDADOS NÃO ENCONTRADOS.")
         print("Você DEVE rodar o 'consolidate_data.py' primeiro para criar os arquivos.")
         print("----------------------------------------------------------------------")
         exit()
    
    # Cria o dataset otimizado
    full_dataset = MemmapSequenceDataset(seq_len=SEQ_LEN)
    
    # Divisão: 80% treino, 20% validação
    total_sequences = len(full_dataset)
    train_size = int(0.8 * total_sequences)
    val_size = total_sequences - train_size
    
    train_ds = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_ds = torch.utils.data.Subset(full_dataset, range(train_size, total_sequences))

    # DataLoaders com batch size e workers otimizados
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Inicialização do Modelo
    model = LSTMBehaviorModel(
        input_size=INPUT_SIZE,
        hidden_size=256,
        num_layers=2,
        num_classes=full_dataset.num_classes,
        lr=1e-4
    )

    # Treinador Lightning
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=True
    )

    print("\n🚀 Iniciando treinamento otimizado (Espere a GPU subir para ~90% de uso)...")
    trainer.fit(model, train_loader, val_loader)