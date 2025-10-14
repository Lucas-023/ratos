import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl


# ============================================
# 1. Modelo principal (LightningModule)
# ============================================
class LSTMBehaviorModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # salva hiperparâmetros automaticamente
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # pega último passo temporal
        out = self.fc(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# ============================================
# 2. DataModule (carrega e divide os dados)
# ============================================
class BehaviorDataModule(pl.LightningDataModule):
    def __init__(self, X, y, batch_size=32):
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = TensorDataset(self.X, self.y)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        self.train_ds, self.val_ds = random_split(dataset, [n_train, n_val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


# ============================================
# 3. Exemplo de uso
# ============================================
if __name__ == "__main__":
    # Dados fictícios (você colocaria seus Parquet processados aqui)
    X = torch.randn(500, 10, 8)  # (samples, seq_len, features)
    y = torch.randint(0, 3, (500,))  # 3 classes de comportamento

    data_module = BehaviorDataModule(X, y, batch_size=32)
    model = LSTMBehaviorModel(input_size=8, hidden_size=64, num_layers=2, num_classes=3)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",  # usa GPU se disponível
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, data_module)
