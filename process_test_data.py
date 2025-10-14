import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List

# =========================================================
# CONFIGURAÇÃO DE CAMINHOS
# =========================================================

# 🚨 AJUSTE AQUI: Caminho para a pasta onde estão os arquivos Parquet de TESTE RAW.
# Exemplo: MABe-mouse-behavior-detection/test_tracking/
BASE_PATH_RAW_TEST = Path("MABe-mouse-behavior-detection/test_tracking/AdaptableSnail") 

# 🚨 AJUSTE AQUI: Pasta de saída para os arquivos processados (um por vídeo)
OUTPUT_DIR = BASE_PATH_RAW_TEST.parent / "processed_videos_test_fixed"

# Garante que a pasta de saída exista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# FUNÇÕES DE PROCESSAMENTO (ADAPTADAS DO SEU DATALOADER)
# =========================================================

def load_parquet(file_path: Path) -> pd.DataFrame:
    """Carrega um arquivo Parquet com engine rápida."""
    return pd.read_parquet(file_path, engine="fastparquet")

def expand_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Expande anotações e garante que a coluna 'behavior' exista (mesmo que vazia)."""
    if "behavior" not in df.columns:
        # Cria a coluna 'behavior' caso o arquivo de teste não a tenha, 
        # preenchendo com NaN/None para evitar erro.
        df["behavior"] = None 
    df["behavior"] = df["behavior"].astype("object") # Mantém como object para lidar com None
    return df

def pivot_tracking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma dados de rastreamento longos em formato wide (uma linha por frame).
    Aviso: Assume que 'behavior' é a única coluna de anotação.
    """
    
    # Colunas de rastreamento típicas em formato Long (a serem pivotadas)
    tracking_cols_long = {"mouse_id", "bodypart", "x", "y"}
    
    # Verifica se as colunas necessárias para PIVOTAR existem (formato Long)
    if tracking_cols_long.issubset(df.columns):
        # Estamos no formato Long, PIVOTAMOS:
        df = (
            df.pivot_table(
                index=["video_id", "frame"], 
                columns=["mouse_id", "bodypart"], 
                values=["x", "y"]
            )
            .sort_index(axis=1)
        )
        
        # Flatten: Transforma o MultiIndex em colunas simples: 'x_mouse1_body_center', etc.
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
        df = df.reset_index(level=["video_id", "frame"])
        tqdm.write("    ✅ Pivotagem Long->Wide concluída.")
        
    else:
        # Estamos no formato Wide (como o log indicou)
        # O DF já está wide, mas precisamos garantir que os nomes de colunas X/Y sejam corretos.
        tqdm.write("    ⚠️ Aviso: Colunas de rastreamento ausentes. Assumindo formato Wide.")
        
        # Como o DF já é wide, apenas removemos as colunas de metadados
        metadata_cols = [
            'video_id', 'frame', 'behavior', 
            'mouse1_sex', 'mouse2_sex', # etc. (todas as colunas não-coordenadas)
        ]
        
        # Encontra as colunas de X/Y restantes no DF atual
        coord_cols_raw = [col for col in df.columns if col not in metadata_cols]
        
        # Tentativa de padronizar: se o nome da coluna não terminar em _x ou _y, 
        # ele deve ser renomeado. Ex: 'mouse1_hip_left_x'
        renames = {}
        for col in coord_cols_raw:
            if col.endswith('_x') and not col.startswith('x_'):
                # Exemplo: Renomeia 'mouse1_hip_left_x' para 'x_mouse1_hip_left' 
                # (ou vice-versa, dependendo do que o pipeline_features espera)
                # O seu pipeline espera colunas que terminam em '_x' ou '_y'. 
                # Se as colunas estiverem como 'x_mouse1_hip_left', precisamos renomear
                
                # Assumindo que o formato do seu treino é 'feature_mouse_bodypart_coord' (ex: 'x_mouse1_hip_left')
                # Vamos apenas garantir que as colunas de coordenadas estejam presentes no DF.
                # Se a lista de 'coord_cols' em pipeline_features for vazia, significa que os nomes 
                # não terminam em _x ou _y. Precisamos ver o que o treino espera.
                pass 
                
        # Mantendo simples: se o DF já for Wide, ele deve conter as colunas de coordenadas
        # com os sufixos _x e _y. Se não contiver, o problema está nos nomes das colunas.
        
    # Remove as colunas de metadados antes de retornar, para que o pipeline de features
    # lide apenas com as coordenadas e features.
    
    # Se o formato Wide for o correto, o problema é que os nomes das colunas de
    # coordenada no seu arquivo de teste (WIDE) não terminam em '_x' ou '_y'.
    
    # Vamos fazer a correção no pipeline_features para ser mais flexível
    return df

# No seu arquivo process_test_data.py:

def pipeline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features importantes (velocidade, aceleração, etc.) a partir de coordenadas.
    CORREÇÃO: Procura por colunas que COMECEM com 'x_' ou 'y_'.
    """
    
    # 1. Colunas de Coordenadas
    # Filtra colunas que pareçam ser coordenadas X/Y
    coord_cols = [
        col for col in df.columns 
        if col.startswith('x_') or col.startswith('y_')
    ]
    
    # Exclui colunas que são metadados, mas que podem ter 'x_' ou 'y_' no nome 
    # (Embora improvável após a pivotagem)
    EXCLUSION_KEYWORDS = ['sex', 'id', 'age', 'condition', 'fps', 'duration']
    coord_cols = [col for col in coord_cols if not any(k in col for k in EXCLUSION_KEYWORDS)]

    if not coord_cols:
        tqdm.write("    ❌ ERRO: Nenhuma coluna de coordenada (x_... ou y_...) encontrada para calcular features.")
        return df

    # 2. Preenchimento de NAs (essencial antes do diff)
    # df[coord_cols] = df[coord_cols].fillna(method='ffill').fillna(method='bfill') # Pandas Warning
    df[coord_cols] = df[coord_cols].ffill().bfill() # Versão sem FutureWarning

    # 3. Cálculo de Velocidade (diff)
    vel_cols = [f"vel_{col}" for col in coord_cols]
    df[vel_cols] = df[coord_cols].diff()
    
    # 4. Cálculo de Features Adicionais (Movimento)
    
    # Encontra as bases das coordenadas (Ex: 'mouse1_hip_left')
    base_parts = set(col[2:] for col in coord_cols if col.startswith('x_'))

    calculated_movement_cols = []
    
    for base_part in base_parts:
        x_col = f"x_{base_part}"
        y_col = f"y_{base_part}"
        
        vel_x_col = f"vel_{x_col}"
        vel_y_col = f"vel_{y_col}"
        
        # Garante que as colunas de VELOCIDADE existam
        if vel_x_col in df.columns and vel_y_col in df.columns:
            movement_col_name = f"total_movement_{base_part}"
            df[movement_col_name] = np.sqrt(
                df[vel_x_col]**2 + df[vel_y_col]**2
            )
            calculated_movement_cols.append(movement_col_name)

    # 5. Remove as linhas iniciais com NaN resultantes do diff()
    df = df.dropna(subset=vel_cols, how='all').reset_index(drop=True)
    
    if not calculated_movement_cols:
         tqdm.write("    ⚠️ Aviso: Nenhuma feature de movimento total foi calculada (pode ser intencional se não há pares X/Y).")

    return df


# =========================================================
# LÓGICA PRINCIPAL DE EXECUÇÃO
# =========================================================

# No seu arquivo process_test_data.py:

def process_test_videos(parquet_files: List[Path]):
    """Processa uma lista de arquivos Parquet raw e salva os resultados."""
    # ... (código inicial permanece o mesmo) ...

    for file_path in tqdm(parquet_files, desc="Processando Vídeos de Teste"):
        try:
            # 1. Carrega e Expande Anotações (necessário para a coluna 'behavior')
            df = load_parquet(file_path)
            
            # =======================================================
            # 🚨 CORREÇÃO: INJETAR COLUNAS DE METADADOS AUSENTES 🚨
            # =======================================================
            video_id = file_path.stem 
            
            if 'video_id' not in df.columns:
                df['video_id'] = video_id
                
            if 'frame' not in df.columns:
                # Cria uma coluna 'frame' baseada no índice do DataFrame
                df['frame'] = df.index 
                
            # Garante que as colunas de ID e FRAME estejam no início
            df = df[['video_id', 'frame'] + [col for col in df.columns if col not in ['video_id', 'frame']]]
            
            # =======================================================
            
            df = expand_annotations(df)

            # 2. Pivotagem (agora deve funcionar com 'video_id' e 'frame')
            df = pivot_tracking(df)
            
            # 3. Feature Engineering
            df = pipeline_features(df)

            # 4. Salva o resultado no formato 'processed'
            output_filename = video_id + "_processed.parquet" # Usa o ID injetado
            output_path = OUTPUT_DIR / output_filename
            
            df.to_parquet(output_path, engine='fastparquet', index=False)
            
            tqdm.write(f"  ✅ {file_path.name} processado e salvo em {output_path.name} (Linhas: {len(df)})")

        except Exception as e:
            tqdm.write(f"  ⚠️ ERRO ao processar {file_path.name}: {e}. Pulando este arquivo.")
            continue # Garante que o loop continue mesmo com erro

# ... (código if __name__ == "__main__" permanece o mesmo) ...
if __name__ == "__main__":
    
    # Se você quiser processar APENAS o arquivo que você subiu:
    # 
    # file_to_process = Path("438887472.parquet")
    # files_to_process = [file_to_process]
    
    # Se você quiser processar TODOS os arquivos de teste:
    files_to_process = list(BASE_PATH_RAW_TEST.rglob("*.parquet"))
    
    process_test_videos(files_to_process)
    
    print("\n----------------------------------------------------")
    print("🚀 FASE DE PROCESSAMENTO DE TESTE CONCLUÍDA.")
    print(f"Os arquivos processados estão em: {OUTPUT_DIR.resolve()}")
    print("O próximo passo é rodar a CONSOLIDAÇÃO nesses novos arquivos.")
    print("----------------------------------------------------")