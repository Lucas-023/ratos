import pandas as pd
from pathlib import Path

# Caminho para os dados ORIGINAIS (não os processados)
TRACKING_ROOT = Path("MABe-mouse-behavior-detection/train_tracking")

# Pega o primeiro arquivo que encontrar
files = list(TRACKING_ROOT.rglob("*.parquet"))

if not files:
    print("❌ Nenhum arquivo encontrado em train_tracking.")
else:
    file_path = files[0]
    print(f"📂 Lendo arquivo original: {file_path}")
    
    try:
        df = pd.read_parquet(file_path)
        print(f"\n📊 Dimensões: {df.shape}")
        print("\n📝 LISTA DE TODAS AS COLUNAS (Copie e cole isso na resposta):")
        print("="*60)
        # Imprime todas as colunas ordenadas
        for col in sorted(df.columns):
            print(f"'{col}',")
        print("="*60)
        
        # Amostra de uma linha para ver o formato dos dados
        print("\n👀 Amostra da primeira linha (Colunas de Coordenadas):")
        coord_cols = [c for c in df.columns if 'x' in c or 'y' in c][:10]
        print(df[coord_cols].iloc[0])
        
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")