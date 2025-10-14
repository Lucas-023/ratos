import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Importa o módulo PyTorch Lightning e as definições de tamanho
# 🚨 CORREÇÃO: Importa a classe com o nome CORRETO
from optlstm import LSTMBehaviorModel 
from optlstm import SEQ_LEN, INPUT_SIZE # Reutiliza as constantes definidas lá

# Importa as definições de colunas (necessário para INPUT_SIZE)
# Assumimos que o dataloader.py está acessível
from dataloader import FEATURE_COLUMNS 

# =========================================================
# CONFIGURAÇÃO DE INFERÊNCIA
# =========================================================

# 🚨 AJUSTE AQUI: Caminho para o CHECKPOINT (pesos) do seu melhor modelo treinado
# Você deve encontrar este caminho dentro da pasta 'lightning_logs' após rodar o treino.
# Exemplo: Path("./lightning_logs/version_0/checkpoints/best_model.ckpt")
CHECKPOINT_PATH = Path("lightning_logs/version_6/checkpoints/epoch=9-step=1113270.ckpt") # <--- SUBSTITUA ESTE

# Caminhos dos arquivos de teste consolidados
CONSOLIDATED_TEST_X_PATH = Path("consolidated_TEST_X.npy")
CONSOLIDATED_TEST_Y_PATH = Path("consolidated_TEST_Y.csv") # Usado apenas para saber o frame count

# 🚨 AJUSTE AQUI: Onde salvar as predições finais
OUTPUT_PREDICTIONS_PATH = Path("./test_predictions_final.csv")
BATCH_SIZE = 512 # Pode ser maior para inferência
NUM_WORKERS = 4
# Mapeamento de índices para nomes de comportamento (Se necessário)
# Você deve obter este mapa (target_map) do seu treinamento
# Exemplo: {0: 'grooming', 1: 'investigating', ...}
BEHAVIOR_MAP = {i: f"CLASSE_{i}" for i in range(10)} # <--- AJUSTE ISTO

# =========================================================
# 1. Dataset de Inferência (Lê apenas X)
# =========================================================

class TestInferenceDataset(Dataset):
    def __init__(self, x_file: Path, y_file: Path, seq_len: int):
        self.seq_len = seq_len
        
        # 1. Carrega Labels (Y) Apenas para o Frame Count
        df_y = pd.read_csv(y_file)
        self.total_frames = len(df_y)
        
        # 2. Mapeamento de Features (X) - USANDO np.memmap
        print(f"Carregando features de teste de: {x_file.name}")
        self.features = np.memmap(
            x_file, 
            dtype=np.float32, 
            mode='r', 
            shape=(self.total_frames, INPUT_SIZE) 
        )
        
        self.num_samples = self.total_frames - self.seq_len + 1
        
        print(f"Total de Frames: {self.total_frames}")
        print(f"Total de Janelas de Inferência: {self.num_samples}")

    def __len__(self):
        # Número de janelas válidas
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # idx é o índice da janela (0, 1, 2, ...)
        start = idx
        end = idx + self.seq_len
        
        # Leitura Ultra-Rápida das Features (X)
        X_np_seq = self.features[start:end, :]
        X_seq = torch.from_numpy(X_np_seq)
        
        # Retorna a sequência de features e o índice do frame que está sendo predito (o último da janela)
        frame_index = end - 1
        
        return X_seq, frame_index

# =========================================================
# 2. Lógica Principal de Inferência
# =========================================================

def run_test_inference(checkpoint_path: Path):
    
    if not checkpoint_path.exists():
        print(f"❌ ERRO: Checkpoint não encontrado em {checkpoint_path}")
        print("Verifique e ajuste a variável CHECKPOINT_PATH.")
        return

    # 1. Carregar o Modelo a partir do Checkpoint
    print(f"Carregando modelo a partir de: {checkpoint_path.name}")
    
    # 🚨 CORREÇÃO: Usa o nome correto da classe
    model = LSTMBehaviorModel.load_from_checkpoint(checkpoint_path)
    model.eval() # Coloca o modelo em modo de avaliação
    
    # 2. Configurar o Dataset e DataLoader
    test_dataset = TestInferenceDataset(CONSOLIDATED_TEST_X_PATH, CONSOLIDATED_TEST_Y_PATH, seq_len=SEQ_LEN)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # 3. Executar a Inferência
    print("\nIniciando inferência nos dados de teste...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    all_frame_indices = []
    
    with torch.no_grad():
        for features, frame_indices in tqdm(test_loader, desc="Fazendo Predições"):
            features = features.to(device)
            
            # Forward pass
            logits = model(features)
            
            # Aplica Sigmoid (Multi-Label) e pega o índice da classe mais provável (Max)
            # Se for Multi-Label, o ideal é pegar todas as classes acima de um threshold (0.5)
            # Para este exemplo, pegaremos apenas a classe com o MAIOR score.
            probabilities = torch.sigmoid(logits)
            
            # Obtém a classe predita de maior probabilidade
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_frame_indices.extend(frame_indices.cpu().tolist())

    # 4. Salvar as Predições
    
    output_df = pd.DataFrame({
        'frame': all_frame_indices,
        'predicted_label_index': all_predictions
    })
    
    # Mapeia os índices para nomes de comportamento (opcional, mas recomendado)
    if BEHAVIOR_MAP:
        output_df['predicted_behavior'] = output_df['predicted_label_index'].map(BEHAVIOR_MAP)
    
    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    
    print("\n=======================================================")
    print("✅ INFERÊNCIA CONCLUÍDA!")
    print(f"Frames Preditos: {len(output_df)}")
    print(f"Predições salvas em: {OUTPUT_PREDICTIONS_PATH.resolve()}")
    print("=======================================================")

if __name__ == "__main__":
    
    # 🚨 Lembre-se de AJUSTAR O CHECKPOINT_PATH E O BEHAVIOR_MAP 🚨
    # O BEHAVIOR_MAP deve ser o mesmo target_map gerado durante o treino!
    
    if not CHECKPOINT_PATH.name == "PATH_TO_YOUR_BEST_MODEL.ckpt":
        run_test_inference(CHECKPOINT_PATH)
    else:
        print("\n---------------------------------------------------------")
        print("⚠️ POR FAVOR, AJUSTE AS VARIÁVEIS DE CONFIGURAÇÃO NO INÍCIO DO SCRIPT:")
        print("1. CHECKPOINT_PATH (Caminho para o melhor .ckpt)")
        print("2. BEHAVIOR_MAP (O mapeamento índice -> nome da classe)")
        print("---------------------------------------------------------")