import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from pathlib import Path
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings('ignore')

# =================================================================
# 1. CONFIGURAÇÕES
# =================================================================
# Onde estão os vídeos que você quer testar?
# Se não tiver a pasta test_tracking, pode usar uma subpasta de train_tracking para testar
INPUT_TRACKING_DIR = Path("MABe-mouse-behavior-detection/test_tracking") 

# Se não tiver test_tracking baixado, descomente a linha abaixo para testar com treino:
# INPUT_TRACKING_DIR = Path("MABe-mouse-behavior-detection/train_tracking")

OUTPUT_FILE = "submission_predictions.csv"
MODEL_PATH = "catboost_mouse_model_tuned.cbm" # Ou o nome do seu modelo de 100 iterações
METADATA_PATH = Path("MABe-mouse-behavior-detection/sequence_metadata.csv")

# Colunas que NÃO entram no modelo (apenas identificadores)
METADATA_COLS_TO_EXCLUDE = [
    'frame', 'video_id', 'unique_frame_id', 'lab_id', 
    'behavior', 'behaviors_labeled', 'mouse_key'
]

# =================================================================
# 2. FUNÇÕES DE PROCESSAMENTO (IDÊNTICAS AO TREINO)
# =================================================================

def load_sequence_metadata(path):
    if not path.exists(): return {}
    df = pd.read_csv(path).set_index('sequence_id')
    return {int(idx): row.to_dict() for idx, row in df.iterrows()}

def convert_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Converte formato Longo para Largo (igual ao treino)."""
    required = ['bodypart', 'x', 'y', 'mouse_id']
    if not all(c in df.columns for c in required): return df

    df['mouse_key'] = 'mouse' + df['mouse_id'].astype(str)
    pivot = df.pivot_table(index='video_frame', columns=['mouse_key', 'bodypart'], values=['x', 'y'])
    
    new_cols = [f"{m}_{p}_{v}" for v, m, p in pivot.columns]
    pivot.columns = new_cols
    pivot = pivot.reset_index().rename(columns={'video_frame': 'frame'})
    return pivot

def pipeline_feature_engineering_robust(df: pd.DataFrame) -> pd.DataFrame:
    """Gera exatamente as mesmas 200+ features do treino (Versão Corrigida)."""
    if df.empty: return pd.DataFrame()
    df = df.copy()
    
    # Tratamento básico
    coord_cols = [c for c in df.columns if c.endswith('_x') or c.endswith('_y')]
    for col in coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').interpolate(limit=5)
    
    feats = pd.DataFrame(index=df.index)
    
    # --- CORREÇÃO AQUI ---
    # Verifica se a coluna existe antes de tentar usar .replace()
    if 'pix_per_cm_approx' in df.columns:
        pix_per_cm = df['pix_per_cm_approx'].replace(0, np.nan).fillna(1.0)
    else:
        pix_per_cm = 1.0
    # ---------------------
    
    # 1. Cinemática Individual
    for m in range(1, 5):
        prefix = f"mouse{m}"
        cols_x = [c for c in df.columns if c.startswith(prefix) and c.endswith('_x')]
        cols_y = [c for c in df.columns if c.startswith(prefix) and c.endswith('_y')]
        
        if not cols_x: continue
        
        # Centro do corpo
        cx_col = next((c for c in cols_x if 'body_center' in c), cols_x[0])
        cy_col = next((c for c in cols_y if 'body_center' in c), cols_y[0])
        
        cx, cy = df[cx_col] / pix_per_cm, df[cy_col] / pix_per_cm
        feats[f'{prefix}_x_cm'] = cx
        feats[f'{prefix}_y_cm'] = cy
        
        # Velocidade e Aceleração
        vel = np.sqrt(cx.diff()**2 + cy.diff()**2).fillna(0)
        feats[f'{prefix}_speed_cm'] = vel
        feats[f'{prefix}_accel'] = vel.diff().fillna(0)
        
        # Ângulos
        nose_x = next((c for c in cols_x if 'nose' in c), None)
        tail_x = next((c for c in cols_x if 'tail' in c or 'spine' in c), None)
        if nose_x and tail_x:
            nose_y, tail_y = nose_x.replace('_x', '_y'), tail_x.replace('_x', '_y')
            dx, dy = df[nose_x] - df[tail_x], df[nose_y] - df[tail_y]
            feats[f'{prefix}_angle'] = np.arctan2(dy, dx).fillna(0)
            feats[f'{prefix}_angular_vel'] = feats[f'{prefix}_angle'].diff().fillna(0)

    # 2. Features Sociais
    import itertools
    active = [m for m in range(1, 5) if f'mouse{m}_x_cm' in feats.columns]
    for m1, m2 in itertools.combinations(active, 2):
        p = f"m{m1}_m{m2}"
        x1, y1 = feats[f'mouse{m1}_x_cm'], feats[f'mouse{m1}_y_cm']
        x2, y2 = feats[f'mouse{m2}_x_cm'], feats[f'mouse{m2}_y_cm']
        
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        feats[f'dist_{p}'] = dist
        feats[f'rel_vel_{p}'] = dist.diff().fillna(0)
        
        if f'mouse{m1}_angle' in feats and f'mouse{m2}_angle' in feats:
            ad = np.abs(feats[f'mouse{m1}_angle'] - feats[f'mouse{m2}_angle'])
            feats[f'angle_diff_{p}'] = np.abs((ad + np.pi) % (2*np.pi) - np.pi)

    # 3. Temporais (Lags e Rolling)
    target_cols = [c for c in feats.columns if 'speed' in c or 'dist' in c]
    if len(target_cols) > 20: target_cols = target_cols[:20] 
    
    for c in target_cols:
        for lag in [1, 5, 10]:
            feats[f'{c}_lag{lag}'] = feats[c].shift(lag).fillna(0)
        for w in [5, 15, 30]:
            r = feats[c].rolling(w, min_periods=1)
            feats[f'{c}_mean_{w}'] = r.mean().fillna(0)
            feats[f'{c}_std_{w}'] = r.std().fillna(0)
            if 'speed' in c: feats[f'{c}_max_{w}'] = r.max().fillna(0)

    # Concatena metadados
    meta_cols = [c for c in df.columns if c not in feats.columns and 'mouse' not in c]
    context = ['mouse1_age', 'mouse2_age', 'frames_per_second', 'video_duration_sec']
    for ctx in context:
        if ctx in df.columns: feats[ctx] = df[ctx]
        
    if 'frame' in df.columns: feats['frame'] = df['frame']
    if 'video_id' in df.columns: feats['video_id'] = df['video_id']
    
    return feats

# =================================================================
# 3. PIPELINE DE PREVISÃO
# =================================================================

def run_inference():
    print(f"🚀 Carregando modelo: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        print("❌ Modelo não encontrado! Treine primeiro.")
        return

    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    expected_features = model.feature_names_
    meta_lookup = load_sequence_metadata(METADATA_PATH)
    files = list(INPUT_TRACKING_DIR.rglob("*.parquet"))
    
    # === PARÂMETRO NOVO ===
    # Se a certeza do modelo for menor que isso, consideramos "Vazio"
    PROBABILITY_THRESHOLD = 0.45 
    
    if not files: return

    print(f"📂 Encontrados {len(files)} arquivos.")
    print("⚡ Iniciando inferência com FILTRO DE PROBABILIDADE...")

    results = []
    
    for file_path in tqdm(files, desc="Prevendo"):
        try:
            # 1. Carrega e Prepara (Igual antes)
            df = pd.read_parquet(file_path)
            df = convert_long_to_wide(df)
            
            seq_id = file_path.stem
            try: sid_int = int(seq_id)
            except: sid_int = None
            if sid_int in meta_lookup:
                for k, v in meta_lookup[sid_int].items(): df[k] = v
            
            lab = file_path.parent.name
            df['video_id'] = f"{lab}_{seq_id}"
            if 'frame' not in df.columns: df['frame'] = np.arange(len(df))

            # 2. Pipeline de Features
            df_features = pipeline_feature_engineering_robust(df)
            
            # 3. Alinha colunas
            X = pd.DataFrame(index=df_features.index)
            for col in expected_features:
                if col in df_features.columns:
                    X[col] = df_features[col]
                else:
                    X[col] = 0
            
            cat_features_indices = model.get_cat_feature_indices()
            if len(cat_features_indices) > 0:
                for idx in cat_features_indices:
                    col_name = expected_features[idx]
                    X[col_name] = X[col_name].astype(str).astype('category')

            # === MUDANÇA CRÍTICA AQUI ===
            # Em vez de predict(), usamos predict_proba()
            probs = model.predict_proba(X)
            
            # Pega a classe com maior probabilidade e o valor dessa probabilidade
            max_probs = np.max(probs, axis=1)
            predicted_indices = np.argmax(probs, axis=1)
            classes = model.classes_
            
            final_preds = []
            for prob, idx in zip(max_probs, predicted_indices):
                if prob < PROBABILITY_THRESHOLD:
                    # Se a confiança for baixa, marcamos como "None"
                    final_preds.append("None")
                else:
                    final_preds.append(classes[idx])
            
            # 5. Prepara output
            output_df = pd.DataFrame({
                'video_id': df_features['video_id'],
                'frame': df_features['frame'],
                'predicted_behavior': final_preds # Usamos a lista filtrada
            })
            
            results.append(output_df)

        except Exception as e:
            print(f"⚠️ Erro em {file_path.name}: {e}")
            continue

    if results:
        print(f"\n💾 Consolidando...")
        final_df = pd.concat(results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Previsões salvas em: {OUTPUT_FILE}")
        
        # Mostra quantos viraram "None"
        print("\n📊 Distribuição (incluindo 'None' filtrados):")
        print(final_df['predicted_behavior'].value_counts())
    else:
        print("❌ Nenhuma previsão gerada.")

if __name__ == "__main__":
    run_inference()