import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

# Pastas
base_folder = Path("MABe-mouse-behavior-detection")
tracking_folder = base_folder / "train_tracking"
annotation_folder = base_folder / "train_annotation"
df_meta = pd.read_csv(base_folder / "train.csv")

# Nova pasta de saída
output_folder = base_folder / "processed_videos_final_fixed_features" # Nova pasta para diferenciar
output_folder.mkdir(exist_ok=True)

# ==============================================================================
# Funções Auxiliares
# ==============================================================================

def load_parquet(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path, engine="fastparquet")

def extract_video_id(file_path: Path) -> int:
    return int(file_path.stem)

def pivot_tracking(df_track: pd.DataFrame, n_mice: int = 4) -> pd.DataFrame:
    """
    Transforma tracking em wide com 1 linha por frame, garantindo todas as colunas para até n_mice.
    (Mantida como estava)
    """
    df = df_track.rename(columns={"video_frame": "frame"})
    # Usamos 'pivot' (não 'pivot_table') para evitar problemas de agregação/perda
    df["col_x"] = "mouse" + df["mouse_id"].astype(str) + "_" + df["bodypart"] + "_x"
    df["col_y"] = "mouse" + df["mouse_id"].astype(str) + "_" + df["bodypart"] + "_y"

    try:
        wide_x = df.pivot(index="frame", columns="col_x", values="x")
        wide_y = df.pivot(index="frame", columns="col_y", values="y")
    except ValueError as e:
        print(f"⚠️ Aviso: Duplicatas encontradas no tracking. Usando pivot_table com 'first'. Erro: {e}")
        wide_x = df.pivot_table(index="frame", columns="col_x", values="x", aggfunc="first")
        wide_y = df.pivot_table(index="frame", columns="col_y", values="y", aggfunc="first")

    wide = pd.concat([wide_x, wide_y], axis=1).reset_index()
    
    # 🔹 Adiciona colunas faltantes para garantir os 4 ratos
    # Garante bodyparts para evitar erro em vídeos sem tracking (se houver)
    bodyparts = df["bodypart"].unique() if not df.empty and "bodypart" in df.columns else []
    
    for m in range(1, n_mice + 1):
        for bp in bodyparts:
            for axis in ["x", "y"]:
                col = f"mouse{m}_{bp}_{axis}"
                if col not in wide.columns:
                    wide[col] = pd.NA 

    tracking_cols = ["frame"] + sorted([c for c in wide.columns if c != "frame"])
    return wide[tracking_cols]

def expand_annotations(df_ann: pd.DataFrame) -> pd.DataFrame:
    """Expande intervalos de anotação em uma linha por frame, agregando múltiplos comportamentos em lista.
    (Mantida como estava)
    """
    if df_ann.empty:
        return pd.DataFrame(columns=["frame", "behavior"])

    expanded = []
    for _, row in df_ann.iterrows():
        for f in range(int(row["start_frame"]), int(row["stop_frame"]) + 1):
            expanded.append({"frame": f, "behavior": row["action"]})

    df_exp = pd.DataFrame(expanded)

    # Agrupa múltiplos comportamentos por frame para EVITAR DUPLICAÇÃO DE LINHA
    return (
        df_exp.groupby("frame")["behavior"]
        .apply(list) # Agrega em uma lista de strings
        .reset_index()
    )


# ==============================================================================
# FUNÇÃO PARA ADICIONAR FEATURES (APLICADA POR VÍDEO)
# ==============================================================================

# ==============================================================================
# FUNÇÃO PARA ADICIONAR FEATURES (APLICADA POR VÍDEO) - CORRIGIDA
# ==============================================================================

def add_features_to_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica o tratamento de NaN, normalização e feature engineering a um único vídeo,
    incluindo a criação de flags de máscara ANTES da interpolação.
    """
    if df.empty:
        return df
        
    df = df.copy() # Otimização: Evita PerformanceWarning
    
    # 1. Identificação de Colunas
    coord_cols = [col for col in df.columns 
                  if (col.endswith('_x') or col.endswith('_y')) and col.startswith('mouse')]

    # Tenta converter colunas de tracking para float (para garantir a limpeza inicial)
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') 

    print("   -> Tipos de dados de tracking garantidos como float.")

    # ----------------------------------------------------------------------
    # 🌟 NOVO PASSO: CRIAÇÃO DAS FLAGS DE MÁSCARA (VALIDITY FLAGS)
    # ----------------------------------------------------------------------
    
    # Flags baseadas na coordenada X do centro do corpo.
    # Se X for NaN ou 0.0, assumimos que o tracking era inválido/ausente para o rato.
    
    for m in range(1, 5):
        center_x = f'mouse{m}_body_center_x'
        
        # Cria a flag ANTES de alterar os NaNs ou 0.0
        if center_x in df.columns:
            # A flag será True se a coordenada for NaN OU for 0.0 (o 0.0 costuma ser usado pelo DLC para 'não detectado')
            df[f'mouse{m}_tracking_invalid'] = df[center_x].isna() | (df[center_x] == 0.0)
            print(f"   -> Flag '{f'mouse{m}_tracking_invalid'}' criada.")


    # 2. Tratamento de NaN e 0.0 (Interpolação)
    
    # 2a. Padroniza 0.0 como NaN (Agora a flag já capturou o 0.0)
    for col in coord_cols:
        df[col] = df[col].replace(0.0, np.nan)

    # 2b. Interpolação Linear (Preenchimento das lacunas)
    for col in coord_cols:
        df[col] = df[col].interpolate(method='linear', limit=10, limit_direction='both')
    
    # 2c. ⚠️ Opcional: Preencher NaNs remanescentes que a interpolação não alcançou (limite>10).
    # Se você quiser garantir que NÃO HÁ NaNs remanescentes nas coordenadas:
    # for col in coord_cols:
    #     df[col] = df[col].fillna(method='ffill').fillna(method='bfill')


    # 3. Normalização: Pixels para Centímetros (CM)
    # ... (o resto do seu código de feature engineering segue aqui) ...
    # 4. Criação de Features de Movimento (Velocidade)
    # 5. Criação de Features de Interação (Distância entre Ratos)

    if 'pix_per_cm_approx' in df.columns and not df['pix_per_cm_approx'].isna().all():
        for col in coord_cols:
            col_cm = col.replace('_x', '_cm_x').replace('_y', '_cm_y')
            df[col_cm] = df[col] / df['pix_per_cm_approx']

        # 4. Criação de Features de Movimento (Velocidade)
        print("   -> Calculando velocidade...")
        for m in range(1, 5):
            center_x_cm = f'mouse{m}_body_center_cm_x'
            center_y_cm = f'mouse{m}_body_center_cm_y'
            
            if center_x_cm in df.columns and center_y_cm in df.columns:
                df[f'mouse{m}_delta_x'] = df[center_x_cm].diff()
                df[f'mouse{m}_delta_y'] = df[center_y_cm].diff()
                
                df[f'mouse{m}_speed_cm_per_frame'] = np.sqrt(
                    df[f'mouse{m}_delta_x']**2 + df[f'mouse{m}_delta_y']**2
                )

        # 5. Criação de Features de Interação (Distância entre Ratos)
        center1_x = 'mouse1_body_center_cm_x'
        center1_y = 'mouse1_body_center_cm_y'
        center2_x = 'mouse2_body_center_cm_x'
        center2_y = 'mouse2_body_center_cm_y'
        
        if all(col in df.columns for col in [center1_x, center1_y, center2_x, center2_y]):
            df['dist_m1_m2_cm'] = np.sqrt(
                (df[center1_x] - df[center2_x])**2 +
                (df[center1_y] - df[center2_y])**2
            )
    else:
        print(f"   ⚠️ Aviso: Coluna 'pix_per_cm_approx' ausente ou vazia. Normalização para CM ignorada.")
        
    return df


# ==============================================================================
# LOOP PRINCIPAL DE PROCESSAMENTO (AGORA INCLUINDO FEATURE ENGINEERING)
# ==============================================================================

for lab_folder in tracking_folder.iterdir():
    if not lab_folder.is_dir():
        continue

    print(f"\n📂 Processando laboratório {lab_folder.name}...")

    tracking_files = list(lab_folder.glob("*.parquet"))
    annotation_lab_folder = annotation_folder / lab_folder.name
    annotation_files = list(annotation_lab_folder.glob("*.parquet"))

    videoid_to_annotation = {extract_video_id(f): load_parquet(f) for f in annotation_files}

    for track_file in tracking_files:
        vid = extract_video_id(track_file)
        print(f"   ⏳ Iniciando vídeo {vid}...")
        df_track_raw = load_parquet(track_file)
        
        # 1. Pivotar tracking
        df_track = pivot_tracking(df_track_raw, n_mice=4)

        # 2. Expandir annotations
        df_ann = videoid_to_annotation.get(
            vid, pd.DataFrame(columns=["start_frame", "stop_frame", "action"])
        )
        df_ann_expanded = expand_annotations(df_ann)

        # 3. Merge tracking + annotations
        df_merged = pd.merge(df_track, df_ann_expanded, on="frame", how="left")

        # 4. Adicionar metadata
        meta_row = df_meta[df_meta["video_id"] == vid]
        if not meta_row.empty:
            meta_dict = meta_row.iloc[0].to_dict()
            for col, val in meta_dict.items():
                if col not in df_merged.columns:
                    df_merged[col] = val

        # 5. GARANTIR UMA LINHA PARA CADA FRAME (0 até MAX)
        if not df_merged.empty:
            all_frames = pd.DataFrame(
                {"frame": range(int(df_merged["frame"].min()), int(df_merged["frame"].max()) + 1)}
            )
            df_full = all_frames.merge(df_merged, on="frame", how="left")
        else:
            df_full = df_merged.copy() 
            
        # 6. 🌟 NOVO PASSO: Aplicar Feature Engineering e Limpeza
        df_final = add_features_to_video(df_full)

        # 7. Salva arquivo processado individual
        out_path = output_folder / f"{vid}_processed_features.parquet"
        df_final.to_parquet(out_path, index=False)
        print(f"   ✅ Vídeo {vid} processado, com features e salvo em {out_path} (Linhas: {len(df_final)})")