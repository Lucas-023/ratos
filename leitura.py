import pandas as pd
from pathlib import Path
from typing import Union

# Pastas
base_folder = Path("MABe-mouse-behavior-detection")
tracking_folder = base_folder / "train_tracking"
annotation_folder = base_folder / "train_annotation"
df_meta = pd.read_csv(base_folder / "train.csv")

# Nova pasta de saída
output_folder = base_folder / "processed_videos_final_fixed" # Nova pasta para garantir limpeza
output_folder.mkdir(exist_ok=True)

def load_parquet(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path, engine="fastparquet")

def extract_video_id(file_path: Path) -> int:
    return int(file_path.stem)

def pivot_tracking(df_track: pd.DataFrame, n_mice: int = 4) -> pd.DataFrame:
    """
    Transforma tracking em wide com 1 linha por frame, garantindo todas as colunas para até n_mice.
    """
    df = df_track.rename(columns={"video_frame": "frame"})
    # 🚨 Usamos 'pivot' (não 'pivot_table') para evitar problemas de agregação/perda
    df["col_x"] = "mouse" + df["mouse_id"].astype(str) + "_" + df["bodypart"] + "_x"
    df["col_y"] = "mouse" + df["mouse_id"].astype(str) + "_" + df["bodypart"] + "_y"

    # Tentamos o pivot. Se falhar, é porque o 'df_track' original tinha duplicatas por frame/mouse/bodypart
    try:
        wide_x = df.pivot(index="frame", columns="col_x", values="x")
        wide_y = df.pivot(index="frame", columns="col_y", values="y")
    except ValueError as e:
        # Se houver duplicatas no df_track_raw (raro, mas possível), usamos pivot_table com first.
        print(f"⚠️ Aviso: Duplicatas encontradas no tracking. Usando pivot_table com 'first'. Erro: {e}")
        wide_x = df.pivot_table(index="frame", columns="col_x", values="x", aggfunc="first")
        wide_y = df.pivot_table(index="frame", columns="col_y", values="y", aggfunc="first")

    wide = pd.concat([wide_x, wide_y], axis=1).reset_index()
    
    # 🔹 Adiciona colunas faltantes para garantir os 4 ratos
    bodyparts = df["bodypart"].unique()
    
    for m in range(1, n_mice + 1):
        for bp in bodyparts:
            for axis in ["x", "y"]:
                col = f"mouse{m}_{bp}_{axis}"
                if col not in wide.columns:
                    wide[col] = pd.NA 

    tracking_cols = ["frame"] + sorted([c for c in wide.columns if c != "frame"])
    return wide[tracking_cols]

def expand_annotations(df_ann: pd.DataFrame) -> pd.DataFrame:
    """Expande intervalos de anotação em uma linha por frame, agregando múltiplos comportamentos em lista."""
    if df_ann.empty:
        return pd.DataFrame(columns=["frame", "behavior"])

    expanded = []
    for _, row in df_ann.iterrows():
        for f in range(int(row["start_frame"]), int(row["stop_frame"]) + 1):
            expanded.append({"frame": f, "behavior": row["action"]})

    df_exp = pd.DataFrame(expanded)

    # 🚨 Agrupa múltiplos comportamentos por frame para EVITAR DUPLICAÇÃO DE LINHA
    return (
        df_exp.groupby("frame")["behavior"]
        .apply(list) # Agrega em uma lista de strings
        .reset_index()
    )

# Itera sobre laboratórios
"""for lab_folder in tracking_folder.iterdir():
    if not lab_folder.is_dir():
        continue

    print(f"📂 Processando laboratório {lab_folder.name}...")

    tracking_files = list(lab_folder.glob("*.parquet"))
    annotation_lab_folder = annotation_folder / lab_folder.name
    annotation_files = list(annotation_lab_folder.glob("*.parquet"))

    videoid_to_annotation = {extract_video_id(f): load_parquet(f) for f in annotation_files}

    for track_file in tracking_files:
        vid = extract_video_id(track_file)
        df_track_raw = load_parquet(track_file)
        
        # 1. Pivotar tracking (1 linha = 1 frame, com todos os 4 ratos garantidos)
        df_track = pivot_tracking(df_track_raw, n_mice=4)

        # 2. Expandir annotations (1 linha = 1 frame, comportamentos em lista)
        df_ann = videoid_to_annotation.get(
            vid, pd.DataFrame(columns=["start_frame", "stop_frame", "action"])
        )
        df_ann_expanded = expand_annotations(df_ann)

        # 3. Merge tracking + annotations
        # O merge aqui é 1:1, pois df_track e df_ann_expanded têm no máximo 1 linha por frame.
        df_merged = pd.merge(df_track, df_ann_expanded, on="frame", how="left")

        # 4. Adicionar metadata
        meta_row = df_meta[df_meta["video_id"] == vid]
        if not meta_row.empty:
            # Garante que os metadados sejam adicionados como colunas constantes
            meta_dict = meta_row.iloc[0].to_dict()
            for col, val in meta_dict.items():
                if col not in df_merged.columns:
                    df_merged[col] = val

        # 5. GARANTIR UMA LINHA PARA CADA FRAME (0 até MAX)
        if not df_merged.empty:
            # Se o tracking existir, usa o range do frame min ao frame max
            all_frames = pd.DataFrame(
                {"frame": range(int(df_merged["frame"].min()), int(df_merged["frame"].max()) + 1)}
            )
            df_full = all_frames.merge(df_merged, on="frame", how="left")
        else:
            # Caso o vídeo seja vazio (improvável)
            df_full = df_merged.copy() 

        # Salva arquivo processado individual
        out_path = output_folder / f"{vid}_processed.parquet"
        df_full.to_parquet(out_path, index=False)
        print(f"✅ Vídeo {vid} processado e salvo em {out_path} (Linhas: {len(df_full)})")"""




# Caminho para a pasta de saída
"""output_folder = Path("MABe-mouse-behavior-detection") / "processed_videos_final_fixed"

# Escolher um arquivo processado (ex: o primeiro da pasta)
parquet_files = list(output_folder.glob("*.parquet"))
print("Arquivos disponíveis:", parquet_files[:5])  # mostra alguns nomes

# Carregar o primeiro para visualizar
df = pd.read_parquet(parquet_files[1], engine="fastparquet")

# Mostrar infos básicas
print("\n📊 Shape:", df.shape)
print(df.head(10))  # mostra as primeiras linhas"""


from typing import Union

# Pasta onde os arquivos limpos estão
INPUT_FOLDER = Path("MABe-mouse-behavior-detection") / "processed_videos_final_fixed"

def merge_parquets(folder_path: Union[str, Path], max_files: int = 0) -> pd.DataFrame:
    """Carrega e combina todos os arquivos .parquet em uma pasta."""
    folder = Path(folder_path)
    parquet_files = list(folder.glob("*.parquet"))
    
    if not parquet_files:
        print(f"⚠️ Aviso: Nenhuma arquivo .parquet encontrado em {folder_path}. Retornando DataFrame vazio.")
        return pd.DataFrame()
    
    if max_files > 0:
        parquet_files = parquet_files[:max_files]
        print(f"Processando os primeiros {len(parquet_files)} arquivos...")
    else:
        print(f"Processando um total de {len(parquet_files)} arquivos...")
        
    all_data = []
    
    for file in parquet_files:
        try:
            df = pd.read_parquet(file, engine='fastparquet')
            all_data.append(df)
        except Exception as e:
            print(f"❌ Erro ao ler o arquivo {file.name}: {e}")
            
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        print(f"✅ Concatenação concluída! DataFrame final com {len(merged_df)} linhas.")
        return merged_df
    else:
        return pd.DataFrame()

# ==============================================================================
# EXECUÇÃO DA CONCATENAÇÃO
# ==============================================================================

print(f"\n--- Iniciando a Combinação Final dos {INPUT_FOLDER.name} ---")

# Carrega e combina TODOS os arquivos da pasta
df_final_dataset = merge_parquets(INPUT_FOLDER)

# Se o merge for bem-sucedido, faz a verificação final de unicidade
if not df_final_dataset.empty:
    print("\n--- Verificação de Duplicação Global ---")
    
    # Cria um ID ÚNICO GLOBAL (video_id + frame)
    df_final_dataset['unique_frame_id'] = (
        df_final_dataset['video_id'].astype(str) + 
        '_' + 
        df_final_dataset['frame'].astype(str)
    )

    # Conta as duplicatas no ID GLOBAL
    duplicates = df_final_dataset['unique_frame_id'].value_counts()
    duplicates = duplicates[duplicates > 1]
    
    print(f"Total de linhas no dataset combinado: {len(df_final_dataset)}")
    
    if duplicates.empty:
        print("✅ SUCESSO: Nenhuma duplicação de linha (video_id + frame) encontrada no dataset combinado.")
    else:
        print(f"❌ ERRO GRAVE: {len(duplicates)} linhas ainda estão duplicadas. Algo no source data está incorreto.")
        print(duplicates.head())



import numpy as np

# A variável df_final_dataset deve estar carregada aqui, resultante do seu merge_parquets.

# --------------------------------------------------------------------------------
# 1. Identificação de Colunas
# --------------------------------------------------------------------------------

# Encontra TODAS as colunas de coordenadas (_x, _y)
coord_cols = [col for col in df_final_dataset.columns 
              if (col.endswith('_x') or col.endswith('_y')) and col.startswith('mouse')]

# --------------------------------------------------------------------------------
# 2. Tratamento de NaN e 0.0 (Interpolação) - CORRIGIDO com .transform()
# --------------------------------------------------------------------------------

print("Iniciando tratamento de NaNs e 0.0...")

# 2a. Padroniza 0.0 como NaN nas coordenadas (Assume que 0.0 significa 'não detectado')
for col in coord_cols:
    df_final_dataset[col] = df_final_dataset[col].replace(0.0, np.nan)

# 2b. Interpolação Linear (AGORA USANDO .transform() para garantir alinhamento de índice)
for col in coord_cols:
    df_final_dataset[col] = df_final_dataset.groupby('video_id')[col].transform(
        lambda x: x.interpolate(method='linear', limit=10, limit_direction='both')
    )

print("✅ Interpolação de tracking concluída.")

# --------------------------------------------------------------------------------
# 3. Normalização: Pixels para Centímetros (CM)
# --------------------------------------------------------------------------------

print("Iniciando conversão de Pixels para CM...")

for col in coord_cols:
    col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
    # Divide a coordenada (em pixels) pelo fator de conversão (pix_per_cm_approx)
    df_final_dataset[col_cm] = df_final_dataset[col] / df_final_dataset['pix_per_cm_approx']

tracking_cols_cm = [col for col in df_final_dataset.columns if col.endswith('_cm_x') or col.endswith('_cm_y')]


# --------------------------------------------------------------------------------
# 4. Criação de Features de Movimento (Velocidade)
# --------------------------------------------------------------------------------

print("Iniciando cálculo de Velocidade...")

for m in range(1, 5):
    center_x_cm = f'mouse{m}_body_center_cm_x'
    center_y_cm = f'mouse{m}_body_center_cm_y'
    
    # Calcula a diferença de X (delta X) e Y (delta Y) entre frames
    df_final_dataset[f'mouse{m}_delta_x'] = df_final_dataset.groupby('video_id')[center_x_cm].diff()
    df_final_dataset[f'mouse{m}_delta_y'] = df_final_dataset.groupby('video_id')[center_y_cm].diff()
    
    # Calcula a velocidade (distância euclidiana da mudança: sqrt(dx² + dy²))
    df_final_dataset[f'mouse{m}_speed_cm_per_frame'] = np.sqrt(
        df_final_dataset[f'mouse{m}_delta_x']**2 + df_final_dataset[f'mouse{m}_delta_y']**2
    )

# --------------------------------------------------------------------------------
# 5. Criação de Features de Interação (Distância entre Ratos)
# --------------------------------------------------------------------------------

print("Iniciando cálculo de Distância Social...")

# Distância entre Mouse 1 e Mouse 2 (usando o centro do corpo)
center1_x = 'mouse1_body_center_cm_x'
center1_y = 'mouse1_body_center_cm_y'
center2_x = 'mouse2_body_center_cm_x'
center2_y = 'mouse2_body_center_cm_y'

df_final_dataset['dist_m1_m2_cm'] = np.sqrt(
    (df_final_dataset[center1_x] - df_final_dataset[center2_x])**2 +
    (df_final_dataset[center1_y] - df_final_dataset[center2_y])**2
)

print("✅ Feature Engineering concluído!")
print(f"Novo Shape do Dataset: {df_final_dataset.shape}")

print(df_final_dataset)