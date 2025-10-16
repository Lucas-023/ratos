import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Tuple, List, Any
from tqdm import tqdm
import os 

# --- IMPORTAÇÕES (ASSUMINDO SEUS ARQUIVOS CONSOLIDADOS SÃO COMPATÍVEIS) ---
try:
    from dataloader import FEATURE_COLUMNS, TARGET_COLUMN 
except ImportError:
    print("⚠️ Aviso: dataloader.py não encontrado. Usando definições mockadas.")
    # Use sua lista REAL de 117 features aqui se o dataloader.py não existir
    FEATURE_COLUMNS = [f'x_{i}' for i in range(1, 118)] 
    TARGET_COLUMN = 'behavior'

# ============================================================
# CONFIGURAÇÕES DO DATASET CONSOLIDADO
# ============================================================
CONSOLIDATED_X_PATH = "consolidated_X.npy"
CONSOLIDATED_Y_PATH = "consolidated_Y.csv"
# Arquivos de estatísticas necessários para verificação
X_MEAN_PATH = "X_mean.npy"
X_STD_PATH = "X_std.npy"

SEQ_LEN = 30
INPUT_SIZE = len(FEATURE_COLUMNS) # 117
BATCH_SIZE = 256 
NUM_WORKERS = os.cpu_count() if os.name != 'nt' else 0 


# ============================================================
# 1. Dataset Otimizado com NumPy Memmap (COM CÁLCULO DE PESO)
# ============================================================

class MemmapSequenceDataset(Dataset):
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        
        # 1. Carrega Labels (Y)
        print(f"Carregando labels (Y) de {CONSOLIDATED_Y_PATH}...")
        df_y = pd.read_csv(CONSOLIDATED_Y_PATH)
        self.Y_labels = df_y.iloc[:, 0].tolist() 
        
        self.total_frames = len(self.Y_labels) 
        self.n_features = INPUT_SIZE          
        
        # 2. Mapeamento de Features (X) - USANDO np.memmap
        print(f"Carregando features (X) de {CONSOLIDATED_X_PATH} via Memmap...")
        self.X_memmap = np.memmap(
            CONSOLIDATED_X_PATH, 
            dtype=np.float32, 
            mode='r', 
            shape=(self.total_frames, self.n_features) 
        )
        
        # 3. Mapeamento de Comportamentos
        self.target_map = self._build_target_map()
        self.num_classes = len(self.target_map)
        
        # 🚨 PONTO CRÍTICO 1: Cálculo e exposição do pos_weight para o modelo 🚨
        self.pos_weight = self._calculate_pos_weight()
        
        # Índices válidos
        self.valid_indices = np.arange(0, self.total_frames - self.seq_len)

        print(f"✅ Total de Sequências para Treino/Val: {len(self.valid_indices)}")
        print(f"✅ Total de Classes de Comportamento: {self.num_classes}")

    def safe_extract_labels(self, label_raw: Any) -> List[str]:
        """Função auxiliar para extrair labels de forma segura, idêntica ao consolidate_data.py."""
        if isinstance(label_raw, (list, np.ndarray, pd.Series)):
             return [str(l).strip() for l in label_raw if str(l).strip()]
        if pd.isna(label_raw): 
            return []
        label_str = str(label_raw).strip()
        if not label_str or label_str.lower() in ('nan', '0.0', '0'):
            return []
        return [l.strip() for l in label_str.split(';') if l.strip()]

    def _build_target_map(self):
        """Reconstrói o mapa de targets."""
        target_map = {}
        label_counter = 0
        
        for label_raw in tqdm(self.Y_labels, desc="Mapeando Labels", leave=False):
            labels_to_process = self.safe_extract_labels(label_raw)
            for label in labels_to_process:
                if label and label not in target_map:
                    target_map[label] = label_counter
                    label_counter += 1
        return target_map

    def _calculate_pos_weight(self):
        """Calcula o pos_weight (Negativos / Positivos) para BCEWithLogitsLoss."""
        num_classes = len(self.target_map)
        pos_counts = np.zeros(num_classes, dtype=np.float64)
        total_frames = self.total_frames
        
        print("📊 Calculando Pesos de Perda (pos_weight)...")
        # Itera sobre todos os labels para contar ocorrências positivas
        for label_raw in tqdm(self.Y_labels, desc="Contando Ocorrências", leave=False):
            labels_present = self.safe_extract_labels(label_raw)
            for label_name in labels_present:
                if label_name in self.target_map:
                    idx = self.target_map[label_name]
                    pos_counts[idx] += 1
        
        # Adiciona 1 para classes com 0 ocorrências (evita divisão por zero/infinito)
        pos_counts_safe = pos_counts + 1 
        
        neg_counts = total_frames - pos_counts_safe
        
        # pos_weight = (Negativos / Positivos). Quanto mais raro, maior o peso.
        pos_weight = neg_counts / pos_counts_safe
        
        print(f"✅ Pesos Calculados. Exemplo de Peso (Classe mais rara): {np.max(pos_weight):.2f}")
        return torch.from_numpy(pos_weight).float()


    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx = self.valid_indices[idx] 
        
        # 1. Leitura Ultra-Rápida das Features (X)
        X_np_seq = self.X_memmap[start_idx : start_idx + self.seq_len]
        
        # 🚨 CORREÇÃO DE AVISO: Usa .copy() para garantir que o tensor seja gravável
        X_seq = torch.from_numpy(X_np_seq.copy()) 
        
        # 2. Processamento do Target (último frame da sequência)
        Y_label_raw = self.Y_labels[start_idx + self.seq_len - 1]
        labels_present = self.safe_extract_labels(Y_label_raw)
        
        # 3. Multi-Hot Encoding do Target (y)
        y_multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        
        for label in labels_present:
            if label and label in self.target_map:
                y_multi_hot[self.target_map[label]] = 1.0 

        return X_seq, y_multi_hot


# ============================================================
# 2. Modelo LSTM (Com uso de pos_weight)
# ============================================================

class LSTMBehaviorModel(pl.LightningModule):
    # 🚨 PONTO CRÍTICO 2: Recebe o pos_weight_tensor no __init__
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr=1e-6, pos_weight_tensor=None):
        super().__init__()
        # O pos_weight é um tensor PyTorch, devemos ignorá-lo no save_hyperparameters
        self.save_hyperparameters(ignore=['pos_weight_tensor']) 
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 🚨 PONTO CRÍTICO 3: Instancia BCEWithLogitsLoss com o pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

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
        # Mantenha o LR baixo para estabilidade (1e-6)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================================
# 3. Script principal (Execução e Ponte)
# ============================================================

if __name__ == "__main__":

    # Verifica se os arquivos consolidados e estatísticos existem
    if not all(Path(p).exists() for p in [CONSOLIDATED_X_PATH, CONSOLIDATED_Y_PATH, X_MEAN_PATH, X_STD_PATH]):
         print("----------------------------------------------------------------------")
         print("❌ ARQUIVOS CONSOLIDADOS OU ESTATÍSTICAS NÃO ENCONTRADOS.")
         print("Você DEVE rodar o 'consolidate_data.py' modificado primeiro.")
         print("----------------------------------------------------------------------")
         exit()
    
    # Cria o dataset otimizado (ELE CALCULA O POS_WEIGHT AQUI)
    full_dataset = MemmapSequenceDataset(seq_len=SEQ_LEN)
    
    # Divisão: 80% treino, 20% validação
    total_sequences = len(full_dataset)
    train_size = int(0.8 * total_sequences)
    val_size = total_sequences - train_size
    
    train_ds = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_ds = torch.utils.data.Subset(full_dataset, range(train_size, total_sequences))

    # DataLoaders 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Inicialização do Modelo (A PONTE)
    # 🚨 PONTO CRÍTICO 4: Passa o pos_weight CALCULADO do Dataset para o Modelo
    model = LSTMBehaviorModel(
        input_size=INPUT_SIZE,
        hidden_size=128, # Reduzido para estabilidade
        num_layers=1,    # Reduzido para estabilidade
        num_classes=full_dataset.num_classes,
        lr=1e-6,         # LR ultra-baixo para estabilidade
        pos_weight_tensor=full_dataset.pos_weight # PASSA O PESO CALCULADO
    )

    # Treinador Lightning
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        enable_checkpointing=True,
        # Configurações de Máxima Estabilidade
        gradient_clip_val=1.0, 
        gradient_clip_algorithm="norm" 
    )

    print("\n🚀 Iniciando treinamento com Ponderação de Perda (Deve ser estável!)...")
    trainer.fit(model, train_loader, val_loader)