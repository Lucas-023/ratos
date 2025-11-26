import pandas as pd
import numpy as np
from tqdm import tqdm

# =================================================================
# CONFIGURAÇÕES
# =================================================================
INPUT_PREDICTIONS = "submission_predictions.csv"  # Arquivo gerado pelo predict_catboost.py
OUTPUT_SUBMISSION = "submission.csv"              # Arquivo final para upload

# Parâmetros de Pós-Processamento (Ajuste fino para melhorar o F-Score)
MIN_EVENT_DURATION = 5   # Eventos menores que 5 frames são removidos (ruído)
MAX_GAP_TO_FILL = 10     # Buracos menores que 10 frames entre eventos iguais são preenchidos
                         # (Ex: Ataque -> Nada -> Nada -> Ataque vira um único Ataque longo)

# Mapeamento de Agentes (Padrão MABe costuma ser mouse1 agindo sobre mouse2)
# Ajuste conforme a tarefa específica da competição se houver variações
DEFAULT_AGENT = 'mouse1'
DEFAULT_TARGET = 'mouse2'

def fill_gaps(series, gap_size):
    """Preenche lacunas pequenas (NaN ou None) entre eventos iguais."""
    # Transforma 'None' ou string vazia em NaN para o pandas tratar
    s = series.replace({'None': np.nan, 'nan': np.nan, '': np.nan})
    
    # Forward fill (preenche pra frente) limitado pelo gap_size
    s_ffill = s.fillna(method='ffill', limit=gap_size)
    
    # Backward fill (preenche pra trás) para garantir que buracos no meio sejam fechados
    s_bfill = s.fillna(method='bfill', limit=gap_size)
    
    # Combina: Só preenche se ambos os lados concordarem ou se for um buraco no meio de iguais
    # Lógica simplificada: usa ffill onde bfill também tem valor (buraco fechado)
    return s_ffill.where(s_ffill == s_bfill, s)

def filter_short_events(df, min_duration):
    """Remove eventos que duram menos que min_duration frames."""
    # Cria grupos de eventos consecutivos
    df['group'] = (df['predicted_behavior'] != df['predicted_behavior'].shift()).cumsum()
    
    # Conta tamanho de cada grupo
    counts = df.groupby('group')['frame'].transform('count')
    
    # Se o grupo for pequeno, substitui por NaN (ou 'other')
    mask_short = counts < min_duration
    df.loc[mask_short, 'predicted_behavior'] = np.nan
    
    return df

def frames_to_intervals(df_video):
    """Converte frames contínuos em intervalos (start, stop)."""
    events = []
    
    # Remove NaNs resultantes da filtragem
    df_clean = df_video.dropna(subset=['predicted_behavior'])
    
    if df_clean.empty:
        return events
    
    # Detecta mudanças de comportamento
    # shift(1) pega o valor anterior. Se for diferente do atual, é um novo evento.
    df_clean['prev_behavior'] = df_clean['predicted_behavior'].shift(1)
    df_clean['frame_diff'] = df_clean['frame'].diff()
    
    # Novo evento se: Comportamento mudou OU Frame não é consecutivo (buraco nos dados)
    df_clean['new_event'] = (df_clean['predicted_behavior'] != df_clean['prev_behavior']) | (df_clean['frame_diff'] > 1)
    
    # Atribui um ID para cada evento
    df_clean['event_id'] = df_clean['new_event'].cumsum()
    
    # Agrupa por evento e pega start/stop
    grouped = df_clean.groupby(['event_id', 'predicted_behavior'])
    
    for (evt_id, action), group in grouped:
        start = int(group['frame'].min())
        stop = int(group['frame'].max())
        
        # Ignora classe 'other' ou 'nan' se o modelo previu isso
        if action.lower() in ['other', 'none', 'nan']:
            continue
            
        events.append({
            'video_id': df_video['video_id'].iloc[0],
            'agent_id': DEFAULT_AGENT,
            'target_id': DEFAULT_TARGET,
            'action': action,
            'start_frame': start,
            'stop_frame': stop
        })
        
    return events

def main():
    print(f"📂 Lendo previsões brutas: {INPUT_PREDICTIONS}...")
    try:
        df = pd.read_csv(INPUT_PREDICTIONS)
    except FileNotFoundError:
        print("❌ Arquivo não encontrado. Rode o predict_catboost.py primeiro.")
        return

    print("⚡ Iniciando Pós-Processamento (Suavização e Conversão)...")
    
    final_intervals = []
    
    # Processa vídeo por vídeo para não misturar frames
    video_ids = df['video_id'].unique()
    
    for vid in tqdm(video_ids, desc="Processando Vídeos"):
        # Pega dados do vídeo e garante ordenação por frame
        df_vid = df[df['video_id'] == vid].sort_values('frame').copy()
        
        # 1. Preenche Gaps (Suavização)
        # Se frame 10=Attack, 11=NaN, 12=Attack -> 11 vira Attack
        df_vid['predicted_behavior'] = fill_gaps(df_vid['predicted_behavior'], MAX_GAP_TO_FILL)
        
        # 2. Remove eventos curtos (Ruído)
        # Se frame 50=Attack mas 49 e 51 são diferentes -> 50 vira NaN
        df_vid = filter_short_events(df_vid, MIN_EVENT_DURATION)
        
        # 3. Converte para Intervalos
        intervals = frames_to_intervals(df_vid)
        final_intervals.extend(intervals)

    # Cria DataFrame final
    submission_df = pd.DataFrame(final_intervals)
    
    if submission_df.empty:
        print("⚠️ Nenhum evento detectado após filtragem!")
        # Cria CSV vazio com colunas certas para não dar erro de formato
        submission_df = pd.DataFrame(columns=['row_id','video_id','agent_id','target_id','action','start_frame','stop_frame'])
    else:
        # Adiciona row_id obrigatório
        submission_df.insert(0, 'row_id', range(len(submission_df)))
        
        # Ordena colunas conforme sample_submission.csv
        cols = ['row_id', 'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']
        submission_df = submission_df[cols]

    print(f"\n💾 Salvando submissão final: {OUTPUT_SUBMISSION}")
    submission_df.to_csv(OUTPUT_SUBMISSION, index=False)
    
    print("\n📊 Resumo da Submissão:")
    print(f"   Total de Eventos: {len(submission_df)}")
    if not submission_df.empty:
        print("   Ações detectadas:")
        print(submission_df['action'].value_counts().head(10))
    print(f"\n✅ Pronto! Faça o upload de '{OUTPUT_SUBMISSION}' no Kaggle.")

if __name__ == "__main__":
    main()