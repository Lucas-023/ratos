"""
Script para diagnosticar por que os labels não estão sendo encontrados.
Testa um arquivo específico para verificar o fluxo completo.
"""

import pandas as pd
from pathlib import Path
from process_data_catboost import (
    prepare_tracking_dataframe,
    load_annotation_file,
    merge_tracking_and_annotations,
)

# Configurações
TRACKING_ROOT = Path("MABe-mouse-behavior-detection/train_tracking")
ANNOTATIONS_ROOT = Path("MABe-mouse-behavior-detection/train_annotation")
MASTER_ANNOTATIONS_PATH = Path("MABe-mouse-behavior-detection/train_annotations.parquet")
FEATURE_ENGINEERED_ROOT = Path("MABe-mouse-behavior-detection/feature_engineered_data_catboost")

print("=" * 60)
print("DIAGNÓSTICO DE ANOTAÇÕES")
print("=" * 60)

# Encontra um arquivo de exemplo
tracking_files = list(TRACKING_ROOT.rglob("*.parquet"))
if not tracking_files:
    print(f"❌ Nenhum arquivo de tracking encontrado em {TRACKING_ROOT}")
    exit(1)

# Testa o primeiro arquivo
test_file = tracking_files[0]
print(f"\n📂 Testando arquivo: {test_file.name}")
print(f"   Caminho completo: {test_file}")

# Extrai lab_name e sequence_id
try:
    relative_parts = test_file.relative_to(TRACKING_ROOT).parts
    lab_name = relative_parts[0]
except ValueError:
    lab_name = test_file.parent.name

sequence_id = test_file.stem
print(f"   • Lab: {lab_name}")
print(f"   • Sequence ID: {sequence_id}")

# 1. Verifica se o arquivo de tracking existe e tem dados
print("\n1️⃣ Verificando arquivo de tracking...")
try:
    df_tracking = pd.read_parquet(test_file, engine='fastparquet')
    print(f"   ✅ Arquivo carregado: {len(df_tracking)} registros")
    print(f"   • Colunas: {list(df_tracking.columns)[:10]}...")
except Exception as e:
    print(f"   ❌ Erro ao carregar: {e}")
    exit(1)

# 2. Verifica arquivos de anotação
print("\n2️⃣ Verificando arquivos de anotação...")
candidate_files = []
if ANNOTATIONS_ROOT.exists():
    candidate1 = ANNOTATIONS_ROOT / lab_name / f"{sequence_id}.parquet"
    candidate2 = ANNOTATIONS_ROOT / f"{sequence_id}.parquet"
    candidate_files.extend([candidate1, candidate2])
    print(f"   • Candidato 1: {candidate1}")
    print(f"      Existe? {candidate1.exists()}")
    print(f"   • Candidato 2: {candidate2}")
    print(f"      Existe? {candidate2.exists()}")

if MASTER_ANNOTATIONS_PATH.exists():
    print(f"   • Master annotations: {MASTER_ANNOTATIONS_PATH}")
    print(f"      Existe? {MASTER_ANNOTATIONS_PATH.exists()}")

# 3. Tenta carregar anotações
print("\n3️⃣ Carregando anotações...")
annotation_df = load_annotation_file(
    sequence_id=sequence_id,
    lab_name=lab_name,
    annotations_root=ANNOTATIONS_ROOT,
    master_annotations_path=MASTER_ANNOTATIONS_PATH,
)

if annotation_df.empty:
    print("   ⚠️ Nenhuma anotação encontrada!")
else:
    print(f"   ✅ Anotações carregadas: {len(annotation_df)} registros")
    print(f"   • Colunas: {list(annotation_df.columns)}")
    if 'behavior' in annotation_df.columns:
        non_empty = annotation_df['behavior'].notna().sum()
        print(f"   • Registros com behavior: {non_empty}/{len(annotation_df)}")
        if non_empty > 0:
            print(f"   • Exemplos de behavior:")
            for i, val in enumerate(annotation_df['behavior'].dropna().head(5)):
                print(f"      [{i}] {repr(str(val)[:100])}")

# 4. Testa merge
print("\n4️⃣ Testando merge de tracking + anotações...")
df_merged = merge_tracking_and_annotations(df_tracking, annotation_df, verbose=True)

if 'behavior' in df_merged.columns:
    non_empty_merged = df_merged['behavior'].notna().sum()
    print(f"   • Após merge: {non_empty_merged}/{len(df_merged)} registros com behavior")
    if non_empty_merged > 0:
        print(f"   • Exemplos de behavior após merge:")
        for i, val in enumerate(df_merged['behavior'].dropna().head(5)):
            print(f"      [{i}] {repr(str(val)[:100])}")
else:
    print("   ⚠️ Coluna 'behavior' não existe após merge!")

# 5. Verifica arquivo processado
print("\n5️⃣ Verificando arquivo processado (se existir)...")
processed_file = FEATURE_ENGINEERED_ROOT / lab_name / f"{sequence_id}.parquet"
if processed_file.exists():
    print(f"   • Arquivo processado: {processed_file}")
    try:
        df_processed = pd.read_parquet(processed_file, engine='fastparquet')
        print(f"   ✅ Arquivo carregado: {len(df_processed)} registros")
        if 'behavior' in df_processed.columns:
            non_empty_processed = df_processed['behavior'].notna().sum()
            print(f"   • Registros com behavior: {non_empty_processed}/{len(df_processed)}")
            if non_empty_processed == 0:
                print("   ⚠️ PROBLEMA: Arquivo processado tem behavior vazio!")
                print("   • Exemplos de behavior no arquivo processado:")
                for i, val in enumerate(df_processed['behavior'].head(10)):
                    print(f"      [{i}] {repr(str(val))}")
        else:
            print("   ⚠️ Coluna 'behavior' não existe no arquivo processado!")
    except Exception as e:
        print(f"   ❌ Erro ao carregar arquivo processado: {e}")
else:
    print(f"   • Arquivo processado não existe ainda: {processed_file}")

print("\n" + "=" * 60)
print("FIM DO DIAGNÓSTICO")
print("=" * 60)

