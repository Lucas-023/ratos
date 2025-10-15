# format_submission.py

import pandas as pd
from pathlib import Path
from typing import Dict

# =========================================================
# CONFIGURAÇÃO DE CAMINHOS E METADADOS
# =========================================================

# 🚨 AJUSTE AQUI: Seu arquivo de predições
PREDICTIONS_PATH = Path("test_predictions_final.csv")
# 🚨 AJUSTE AQUI: O caminho do arquivo de submissão final
OUTPUT_SUBMISSION_PATH = Path("submission_formatted.csv")

# 🚨 METADADOS DO VÍDEO DE TESTE 🚨
# Estes dados são necessários porque seu CSV de predição só tem 'frame'.
# Como você só tem 1 arquivo de teste processado, vamos codificá-lo:
VIDEO_ID = "438887472" 
AGENT_ID = "mouse1" # Assumindo que a predição é focada em um agente principal
TARGET_ID = "mouse2" # Assumindo um agente alvo (ou 'nan' se a ação for individual)

# 🚨 AJUSTE AQUI: Mapeamento Final de Comportamento 🚨
# O 'CLASSE_0' do seu CSV precisa ser mapeado para o nome real do comportamento (ex: 'grooming', 'sniff').
# O seu BEHAVIOR_MAP do run_inference.py deve ser usado aqui.
BEHAVIOR_MAP_FINAL: Dict[str, str] = {
    "CLASSE_0": "sniff", # <--- AJUSTE ISTO (MUITO IMPORTANTE!)
    "CLASSE_1": "grooming",
    # Adicione todos os seus 10+ comportamentos mapeados aqui
}

# =========================================================
# FUNÇÃO PRINCIPAL DE CONVERSÃO
# =========================================================

def rle_to_submission(df_predictions: pd.DataFrame, video_id: str, agent_id: str, target_id: str) -> pd.DataFrame:
    """
    Converte predições frame-a-frame em um formato de submissão baseado em
    intervalos (start_frame, stop_frame) usando Run-Length Encoding (RLE).
    """
    
    # 1. Renomeia a coluna de ação para simplificar
    df_predictions = df_predictions.rename(columns={'predicted_behavior': 'action'})
    
    # 2. Mapeia as classes 'CLASSE_X' para o nome final
    # Apenas mapeia o que estiver no dicionário, mantendo o original se a chave não existir
    df_predictions['action'] = df_predictions['action'].map(BEHAVIOR_MAP_FINAL).fillna(df_predictions['action'])
    
    # 3. Identifica inícios de novos segmentos de ação (RLE)
    # Cria uma flag 'start_of_run' onde a ação muda OU é o primeiro frame.
    df_predictions['prev_action'] = df_predictions['action'].shift(1)
    df_predictions['start_of_run'] = (df_predictions['action'] != df_predictions['prev_action']) | (df_predictions.index == 0)
    
    # Cria um ID de grupo para cada sequência contínua de ação
    df_predictions['run_id'] = df_predictions['start_of_run'].cumsum()
    
    # 4. Agrupa para encontrar start/stop frames
    submission_df = df_predictions.groupby('run_id').agg(
        # Pega a ação do segmento
        action=('action', 'first'),
        # O start_frame é o frame do primeiro elemento do grupo
        start_frame=('frame', 'min'),
        # O stop_frame é o frame do último elemento do grupo + 1
        stop_frame=('frame', lambda x: x.max() + 1)
    ).reset_index(drop=True)
    
    # 5. Adiciona colunas de metadados
    submission_df['video_id'] = video_id
    submission_df['agent_id'] = agent_id
    submission_df['target_id'] = target_id 
    
    # 6. Reordena e Adiciona row_id
    submission_df = submission_df[[
        'video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'
    ]]
    submission_df.insert(0, 'row_id', submission_df.index)
    
    # 7. FILTRO OPCIONAL: Remove ações de 'nan' ou 'sem_acao' se o formato final não permitir
    # Assumimos que a classe com maior índice (geralmente a classe de 'background' ou 'sem_acao') 
    # não deve ser submetida, mas isto depende da competição.
    # Ex: submission_df = submission_df[submission_df['action'] != 'sem_acao']
    
    print(f"Predições convertidas de {len(df_predictions)} frames para {len(submission_df)} intervalos.")
    
    return submission_df

# =========================================================
# EXECUÇÃO
# =========================================================

if __name__ == "__main__":
    if not PREDICTIONS_PATH.exists():
        print(f"❌ ERRO: Arquivo de predições não encontrado em {PREDICTIONS_PATH}")
        print("Certifique-se de que o run_inference.py foi executado.")
        exit()

    df_preds = pd.read_csv(PREDICTIONS_PATH)
    
    # Verifica se os mapeamentos foram ajustados
    if "CLASSE_0" in BEHAVIOR_MAP_FINAL.values():
         print("\n⚠️ AVISO: O BEHAVIOR_MAP_FINAL não foi ajustado. Usando 'sniff' como padrão.")
         print("Isso PODE resultar em pontuação zero na submissão.")

    df_submission = rle_to_submission(
        df_preds, 
        video_id=VIDEO_ID, 
        agent_id=AGENT_ID, 
        target_id=TARGET_ID
    )
    
    # Salva o resultado final no formato de submissão
    df_submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)
    
    print("\n----------------------------------------------------")
    print("✅ FORMATO DE SUBMISSÃO GERADO!")
    print(f"Arquivo de submissão salvo em: {OUTPUT_SUBMISSION_PATH.resolve()}")
    print("----------------------------------------------------")