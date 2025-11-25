"""
Script de diagnóstico para verificar o formato do arquivo de labels.
Execute antes do train_catboost_example.py para entender problemas.
"""

import pandas as pd
from pathlib import Path

Y_PATH = Path("consolidated_Y_catboost.csv")

print("=" * 60)
print("DIAGNÓSTICO DO ARQUIVO DE LABELS")
print("=" * 60)

if not Y_PATH.exists():
    print(f"❌ Arquivo não encontrado: {Y_PATH}")
    exit(1)

print(f"\n📂 Arquivo: {Y_PATH}")
print(f"   Tamanho: {Y_PATH.stat().st_size / (1024**2):.2f} MB")

# Lê primeiras linhas brutas
print("\n📄 Primeiras 10 linhas brutas do arquivo:")
with open(Y_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        print(f"   [{i}] {repr(line[:200])}")

# Tenta carregar como CSV
print("\n📊 Tentando carregar como CSV...")
try:
    y_df = pd.read_csv(Y_PATH)
    print(f"✅ CSV carregado com sucesso!")
    print(f"   • Shape: {y_df.shape}")
    print(f"   • Colunas: {list(y_df.columns)}")
    
    if len(y_df.columns) == 0:
        print("   ⚠️ DataFrame sem colunas!")
    else:
        # Usa primeira coluna se 'behavior' não existir
        col_name = 'behavior' if 'behavior' in y_df.columns else y_df.columns[0]
        print(f"\n📋 Analisando coluna: '{col_name}'")
        
        # Estatísticas
        total = len(y_df)
        non_empty = y_df[col_name].notna().sum()
        empty = y_df[col_name].isna().sum()
        
        print(f"   • Total de registros: {total:,}")
        print(f"   • Não vazios: {non_empty:,}")
        print(f"   • Vazios/NaN: {empty:,}")
        
        # Mostra exemplos
        print(f"\n📝 Exemplos de valores:")
        sample_values = y_df[col_name].dropna().head(20)
        for i, val in enumerate(sample_values):
            val_str = str(val)
            print(f"   [{i}] tipo={type(val).__name__}, valor={repr(val_str[:100])}")
        
        # Verifica formato
        print(f"\n🔍 Análise de formato:")
        has_semicolon = y_df[col_name].astype(str).str.contains(';', na=False).sum()
        print(f"   • Registros com ';' (separador): {has_semicolon:,}")
        
        # Tenta parsear
        print(f"\n🧪 Testando parse_multi_label:")
        from train_catboost_example import parse_multi_label
        
        labels_found = set()
        for val in y_df[col_name].dropna().head(1000):
            parsed = parse_multi_label(val)
            labels_found.update(parsed)
        
        print(f"   • Labels únicos encontrados (primeiros 1000 registros): {len(labels_found)}")
        if labels_found:
            print(f"   • Primeiros 20 labels: {sorted(list(labels_found))[:20]}")
        else:
            print(f"   ⚠️ NENHUM label encontrado nos primeiros 1000 registros!")
            
except Exception as e:
    print(f"❌ Erro ao carregar CSV: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("FIM DO DIAGNÓSTICO")
print("=" * 60)

