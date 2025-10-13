import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
import fastparquet 
from tqdm import tqdm 

# =================================================================
# 1. Definição de Colunas e Funções Auxiliares
# =================================================================

# Colunas de Features (Coordenadas X, Y das partes do corpo)
FEATURE_COLUMNS = [
    'mouse1_nose_x', 'mouse1_nose_y', 'mouse1_tail_base_x', 'mouse1_tail_base_y', 'mouse1_body_center_x', 'mouse1_body_center_y', 'mouse1_ear_left_x', 'mouse1_ear_left_y', 'mouse1_ear_right_x', 'mouse1_ear_right_y',
    'mouse2_nose_x', 'mouse2_nose_y', 'mouse2_tail_base_x', 'mouse2_tail_base_y', 'mouse2_body_center_x', 'mouse2_body_center_y', 'mouse2_ear_left_x', 'mouse2_ear_left_y', 'mouse2_ear_right_x', 'mouse2_ear_right_y',
    'mouse3_nose_x', 'mouse3_nose_y', 'mouse3_tail_base_x', 'mouse3_tail_base_y', 'mouse3_body_center_x', 'mouse3_body_center_y', 'mouse3_ear_left_x', 'mouse3_ear_left_y', 'mouse3_ear_right_x', 'mouse3_ear_right_y',
    'mouse4_nose_x', 'mouse4_nose_y', 'mouse4_tail_base_x', 'mouse4_tail_base_y', 'mouse4_body_center_x', 'mouse4_body_center_y', 'mouse4_ear_left_x', 'mouse4_ear_left_y', 'mouse4_ear_right_x', 'mouse4_ear_right_y',
]
TARGET_COLUMN = 'behavior' 

def load_parquet_row(file_path: Path, row_index: int, columns: List[str]) -> pd.DataFrame:
    """Carrega uma linha específica de um arquivo Parquet."""
    df = pd.read_parquet(
        file_path, 
        engine='fastparquet', 
        columns=columns, 
    )
    single_row = df.iloc[row_index]
    return single_row

def safe_extract_labels(raw_value: Any) -> List[str]:
    """
    Extrai labels de comportamento de forma segura, lidando com strings, listas,
    arrays NumPy/Pandas e valores NaN, prevenindo o erro de 'ambiguous truth value'.
    """
    labels = []
    
    # Se for um array ou Series (o que causa o erro de ambiguidade), forçamos a iteração
    if isinstance(raw_value, (pd.Series, np.ndarray)):
        for item in raw_value:
            if pd.notna(item) and item is not None:
                labels.append(str(item).strip())
    # Se for uma lista Python normal (esperado para Multi-label)
    elif isinstance(raw_value, list):
        for item in raw_value:
            if pd.notna(item) and item is not None:
                labels.append(str(item).strip())
    # Se for um valor escalar (string, int, float) e não for NaN
    elif pd.notna(raw_value):
        labels.append(str(raw_value).strip())
        
    # Retorna apenas labels não vazias
    return [lbl for lbl in labels if lbl]

# =================================================================
# 2. A Classe LazyFrameDataset (Com Correção Final de Tipo)
# =================================================================

class LazyFrameDataset(Dataset):
    """
    Dataset customizado que lida com o carregamento preguiçoso (lazy loading)
    e com problemas de multi-label (listas na coluna 'behavior').
    """

    def __init__(self, parquet_files: List[Path], features: List[str], target: str):
        self.target_map: dict = {}
        self.features = features
        self.target = target
        self.frame_map: List[Tuple[Path, int]] = []
        
        print("🛠️ Pré-processando metadados e mapeando comportamentos...")
        self._build_frame_map(parquet_files)
        
        print(f"✅ Inicialização concluída. Total de {len(self.frame_map)} frames disponíveis.")
        print(f"Total de classes de comportamento únicas: {len(self.target_map)}")


    def _build_frame_map(self, parquet_files: List[Path]):
        """
        Gera o mapa de frames e o mapa de labels de comportamento.
        Adicionado tqdm para exibir o progresso do I/O.
        """
        label_counter = 0
        
        # ❗ MUDANÇA AQUI: Adiciona tqdm para mostrar o progresso
        for file_path in tqdm(parquet_files, desc="Mapeando Arquivos"):
            try:
                temp_df = pd.read_parquet(
                    file_path, 
                    engine='fastparquet', 
                    columns=['frame', self.target]
                )
                
                num_frames = len(temp_df)
                
                for local_index in range(num_frames):
                    self.frame_map.append((file_path, local_index))
                    
                    # Usando a função segura para extrair e iterar
                    behavior_label = temp_df.iloc[local_index][self.target]
                    labels_to_process = safe_extract_labels(behavior_label)
                    
                    # Mapeia todas as labels encontradas
                    for label in labels_to_process:
                        if label not in self.target_map:
                            self.target_map[label] = label_counter
                            label_counter += 1
                
            except Exception as e:
                # O tqdm ainda mostrará o progresso, mas este erro será impresso
                print(f"\n⚠️ Erro ao pré-processar metadados em {file_path.name}: {e}. Pulando.")
                continue



    def __len__(self) -> int:
        return len(self.frame_map)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        file_path, local_index = self.frame_map[idx]
        cols_to_load = list(set(self.features + [self.target])) 
        num_classes = len(self.target_map)

        try:
            # 1. Carrega a linha específica do disco (Lazy Loading)
            row = load_parquet_row(file_path, local_index, cols_to_load)
            
            # 2. Processamento das Features (X)
            X_df = row[self.features] 
            X_values = X_df.values.astype(np.float32)
            X = torch.tensor(X_values, dtype=torch.float32)
            
            # 3. Processamento do Target (y) - Multi-Hot Encoding
            Y_label_raw = row[self.target]
            labels_present = safe_extract_labels(Y_label_raw)
            
            # Inicializa o vetor multi-hot de zeros
            y_multi_hot = torch.zeros(num_classes, dtype=torch.float32)
            
            for label in labels_present:
                if label in self.target_map:
                    # Se o comportamento estiver mapeado, setamos o índice para 1.0
                    label_id = self.target_map[label]
                    y_multi_hot[label_id] = 1.0 

            # O target agora é um vetor multi-hot (multi-label classification)
            return X, y_multi_hot
            
        except KeyError as e:
            # Erro de coluna (alguma feature pode estar faltando)
            print(f"⚠️ Erro de coluna ({e}) em {file_path.name}. Retornando dummy.")
            dummy_X = torch.zeros(len(self.features), dtype=torch.float32)
            dummy_y = torch.zeros(num_classes, dtype=torch.float32) 
            return dummy_X, dummy_y
        
        except Exception as e:
            # Erros gerais
            # print(f"⚠️ Erro inesperado ao carregar frame {local_index} de {file_path.name}: {e}. Retornando dummy.")
            dummy_X = torch.zeros(len(self.features), dtype=torch.float32)
            dummy_y = torch.zeros(num_classes, dtype=torch.float32)
            return dummy_X, dummy_y

# =================================================================
# 3. Exemplo de Uso
# =================================================================

if __name__ == "__main__":
    # ❗ Altere este caminho: Mude para o diretório raiz dos seus arquivos Parquet
    base_path = Path("MABe-mouse-behavior-detection/processed_videos_final_fixed")
    
    parquet_files = list(base_path.rglob("*.parquet"))

    if not parquet_files:
        print(f"❌ NENHUM arquivo Parquet encontrado. Verifique o caminho.")
    else:
        print(f"🔍 {len(parquet_files)} arquivos Parquet encontrados.")
        
        try:
            mouse_dataset = LazyFrameDataset(
                parquet_files=parquet_files,
                features=FEATURE_COLUMNS,
                target=TARGET_COLUMN
            )
            
            BATCH_SIZE = 64 
            
            data_loader = DataLoader(
                dataset=mouse_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,          
                num_workers=4,         
                pin_memory=True        
            )
            
            print("\n✅ DataLoader criado com sucesso.")
            print(f"Total de batches por época: {len(data_loader)}")

            print("\n▶️ Testando o carregamento de 5 batches...")
            for batch_idx, (X_batch, y_batch) in enumerate(data_loader):
                print(f"Batch {batch_idx+1}:")
                print(f"  - Features (X) shape: {X_batch.shape}")
                print(f"  - Labels (y) shape: {y_batch.shape}")
                
                if batch_idx >= 4:
                    break 

        except Exception as main_e:
            print(f"\n❌ Erro crítico na execução principal: {main_e}")