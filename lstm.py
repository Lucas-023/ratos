import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from dataloader import LazyFrameDataset, FEATURE_COLUMNS, TARGET_COLUMN
from tqdm import tqdm

# ============================================================
# 1. Dataset de Sequências (usando seu LazyFrameDataset base)
# ============================================================

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, seq_len=30):
        self.base_dataset = base_dataset
        self.seq_len = seq_len
        # O frame_map é a lista de todos os frames (Path, local_index)
        self.total_frames = len(self.base_dataset) 

    def __len__(self):
        # O número total de sequências possíveis é o número de frames - seq_len
        return self.total_frames - self.seq_len

    def __getitem__(self, idx):
        try:
            # CORREÇÃO CRÍTICA: Chama a função de leitura em bloco
            # Isso garante UMA ÚNICA leitura de disco para 30 frames.
            X_seq, y = self.base_dataset.get_sequence_data(idx, self.seq_len)
            
            # X_seq já está no formato (seq_len, input_size), pronto para o LSTM
            return X_seq, y

        except (IndexError, ValueError) as e:
            # Este bloco captura sequências que saem dos limites do dataset 
            # ou que cruzam o limite de um arquivo Parquet.
            # Se for um problema de limite, retornamos um tensor zero para evitar falhas,
            # mas o melhor é evitar que esses índices sejam gerados.
            
            # (Opcional, mas mais seguro para não quebrar o num_workers)
            dummy_X = torch.zeros(self.seq_len, len(FEATURE_COLUMNS), dtype=torch.float32)
            dummy_y = torch.zeros(len(self.base_dataset.target_map), dtype=torch.float32)
            return dummy_X, dummy_y


# ============================================================
# 2. Modelo LSTM com PyTorch Lightning
# ============================================================

class LSTMBehaviorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # pega a saída do último frame
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
# 3. Script principal
# ============================================================

if __name__ == "__main__":
    base_path = Path("MABe-mouse-behavior-detection/feature_engineered_data")
    parquet_files = list(base_path.rglob("*.parquet"))

    # Cria o dataset base (frame a frame)
    base_dataset = LazyFrameDataset(parquet_files, features=FEATURE_COLUMNS, target=TARGET_COLUMN)
    num_classes = len(base_dataset.target_map)

    # Cria dataset de sequência
    seq_dataset = SequenceDataset(base_dataset, seq_len=30)

    # Divide em treino e validação
    split = int(0.8 * len(seq_dataset))
    train_ds = torch.utils.data.Subset(seq_dataset, range(0, split))
    val_ds = torch.utils.data.Subset(seq_dataset, range(split, len(seq_dataset)))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=16)

    # Modelo
    model = LSTMBehaviorModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=256,
        num_layers=2,
        num_classes=num_classes,
        lr=1e-4
    )

    # Treinador Lightning
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)
