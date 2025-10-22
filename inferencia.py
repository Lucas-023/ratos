import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json 
import os 

# Importa as definições do seu script de treino
try:
    from optlstm import LSTMBehaviorModel, SEQ_LEN, INPUT_SIZE
except ImportError:
    print("❌ ERRO: Não foi possível importar LSTMBehaviorModel ou constantes de optlstm.py")
    print("Verifique se optlstm.py e suas dependências estão no caminho correto.")
    exit()

# =========================================================
# CONFIGURAÇÃO DE INFERÊNCIA
# =========================================================

# 🚨 AJUSTE AQUI: Caminho para o CHECKPOINT (pesos) do seu melhor modelo treinado
# Certifique-se de que este caminho está correto.
CHECKPOINT_PATH = Path("lightning_logs/version_14/checkpoints/epoch=9-step=1113270.ckpt") 

# Caminhos dos arquivos
CONSOLIDATED_TEST_X_PATH = Path("consolidated_TEST_X.npy")
CONSOLIDATED_TEST_Y_PATH = Path("consolidated_TEST_Y.csv") 
BEHAVIOR_MAP_PATH = Path("behavior_map.json") 

# Configurações de performance/saída
OUTPUT_PREDICTIONS_PATH = Path("./test_predictions_final_ANALYSIS.csv")
BATCH_SIZE = 512 
NUM_WORKERS = os.cpu_count() if os.name != 'nt' else 0
THRESHOLD = 0.5 

BEHAVIOR_MAP = None 

# =========================================================
# 1. Dataset de Inferência (Lê apenas X)
# =========================================================

class TestInferenceDataset(Dataset):
    def __init__(self, x_file: Path, y_file: Path, seq_len: int):
        self.seq_len = seq_len
        
        df_y = pd.read_csv(y_file)
        self.total_frames = len(df_y)
        
        print(f"Carregando features de teste de: {x_file.name}")
        self.features = np.memmap(
            x_file, 
            dtype=np.float32, 
            mode='r', 
            shape=(self.total_frames, INPUT_SIZE) 
        )
        
        self.num_samples = self.total_frames - self.seq_len + 1
        self.output_frame_indices = np.arange(self.seq_len - 1, self.total_frames)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        start = idx
        end = idx + self.seq_len
        
        X_np_seq = self.features[start:end, :]
        X_seq = torch.from_numpy(X_np_seq.copy()) 
        
        frame_index = self.output_frame_indices[idx]
        
        return X_seq, frame_index

# =========================================================
# 2. Lógica Principal de Inferência
# =========================================================

def run_test_inference(checkpoint_path: Path):
    global BEHAVIOR_MAP

    if not checkpoint_path.exists():
        print(f"❌ ERRO: Checkpoint não encontrado em {checkpoint_path}")
        return

    # 1. Carregar o Mapa de Comportamento
    if BEHAVIOR_MAP_PATH.exists():
        with open(BEHAVIOR_MAP_PATH, "r") as f:
            loaded_map = json.load(f)
            BEHAVIOR_MAP = {int(k): v for k, v in loaded_map.items()}
    else:
        print(f"❌ ERRO: Arquivo de mapeamento {BEHAVIOR_MAP_PATH} não encontrado.")
        print("Rode o optlstm.py para gerar behavior_map.json.")
        return

    # 2. Carregar o Modelo a partir do Checkpoint
    print(f"Carregando modelo a partir de: {checkpoint_path.name}")
    
    try:
        # Carrega o checkpoint completo do PyTorch
        checkpoint = torch.load(checkpoint_path, map_location="cpu") 
        hparams = checkpoint["hyper_parameters"]
    except Exception as e:
        print(f"❌ ERRO ao carregar o arquivo checkpoint: {e}")
        return

    # 🚨 PASSO CRÍTICO: Remove a chave problemática (necessário para compatibilidade com checkpoints antigos)
    if "loss_fn.pos_weight" in checkpoint["state_dict"]:
        del checkpoint["state_dict"]["loss_fn.pos_weight"]
        print("✅ Chave 'loss_fn.pos_weight' removida com sucesso do estado de dicionário.")
    
    # 🚨 CORREÇÃO FINAL DO TypeError: Instancia o modelo manualmente e carrega o estado
    
    # Obtém weight_decay, com fallback para 0.0 caso o checkpoint não tenha sido treinado com ele
    weight_decay_val = hparams.get('weight_decay', 0.0) 

    # Instancia o modelo com os hparams salvos (sem 'pos_weight_tensor')
    model = LSTMBehaviorModel(
        input_size=hparams['input_size'],
        hidden_size=hparams['hidden_size'],
        num_layers=hparams['num_layers'],
        num_classes=len(BEHAVIOR_MAP), # Usa o número correto de classes do mapa carregado
        lr=hparams['lr'],
        weight_decay=weight_decay_val  # ✅ NOVO: Adiciona weight_decay
        # pos_weight_tensor removido
    )
    
    # Carrega o estado de dicionário MODIFICADO
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    model.eval() # Coloca o modelo em modo de avaliação
    
    # 3. Configurar o Dataset e DataLoader
    test_dataset = TestInferenceDataset(CONSOLIDATED_TEST_X_PATH, CONSOLIDATED_TEST_Y_PATH, seq_len=SEQ_LEN)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS
    )
    
    # 4. Executar a Inferência e Coletar Probabilidades
    print("\nIniciando inferência e coletando probabilidades...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probabilities = []
    all_frame_indices = []
    
    with torch.no_grad():
        for features, frame_indices in tqdm(test_loader, desc="Fazendo Predições"):
            features = features.to(device)
            logits = model(features)
            probabilities = torch.sigmoid(logits)
            
            all_probabilities.extend(probabilities.cpu().tolist())
            all_frame_indices.extend(frame_indices.cpu().tolist())

    # 5. Salvar as Predições e Probabilidades para Análise
    
    prob_columns = [f"prob_{BEHAVIOR_MAP.get(i, f'CLASSE_{i}')}" for i in range(len(BEHAVIOR_MAP))]
    
    prob_df = pd.DataFrame(all_probabilities, columns=prob_columns)
    
    output_df = pd.DataFrame({'frame': all_frame_indices})
    output_df = pd.concat([output_df, prob_df], axis=1)

    # 1. Predição Multi-Label (Threshold 0.5)
    predicted_multi_label_df = (prob_df > THRESHOLD).astype(int)
    predicted_multi_label_list = predicted_multi_label_df.apply(
        lambda row: ';'.join([col.replace('prob_', '') for col, val in row.items() if val == 1]), axis=1
    )
    output_df[f'predicted_behaviors_multi_label_T{int(THRESHOLD*100)}'] = predicted_multi_label_list
    
    # 2. Predição Single-Label (Classe com maior score)
    output_df['predicted_behavior_argmax'] = prob_df.idxmax(axis=1).apply(lambda x: x.replace('prob_', ''))
    
    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    
    print("\n=======================================================")
    print("✅ INFERÊNCIA CONCLUÍDA E SALVA PARA ANÁLISE!")
    print(f"Predições salvas em: {OUTPUT_PREDICTIONS_PATH.resolve()}")
    print("🔥 AGORA, ANALISE AS COLUNAS 'prob_...' para avaliar o aprendizado.")
    print("=======================================================")

if __name__ == "__main__":
    
    if CHECKPOINT_PATH.name == "PATH_TO_YOUR_BEST_MODEL.ckpt":
        print("\n---------------------------------------------------------")
        print("⚠️ POR FAVOR, AJUSTE A VARIÁVEL CHECKPOINT_PATH NO INÍCIO DO SCRIPT.")
        print("---------------------------------------------------------")
    else:
        run_test_inference(CHECKPOINT_PATH)