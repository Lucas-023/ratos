# calculate_weights.py
import pandas as pd
import numpy as np
import json

# Caminho para o seu CSV de labels consolidado
CONSOLIDATED_Y_PATH = "consolidated_Y.csv" 
BEHAVIOR_MAP_PATH = "behavior_map.json"
WEIGHTS_OUTPUT_PATH = "class_weights.npy"

print("1. Carregando os dados de labels...")
df_y = pd.read_csv(CONSOLIDATED_Y_PATH, header=None, names=['behavior'])

# Remove linhas NaN/vazias que podem ter sido criadas
df_y.dropna(inplace=True)
total_samples = len(df_y)
print(f"Total de amostras (frames) encontradas: {total_samples}")

print("2. Criando o mapa de comportamentos (behavior_map)...")
# Cria uma lista única de todos os comportamentos possíveis
all_labels_set = set()
for labels in df_y['behavior'].str.split(';'):
    all_labels_set.update(labels)

# Remove strings vazias, se houver
all_labels_set.discard('') 

# Ordena para consistência e cria o mapa
behavior_list = sorted(list(all_labels_set))
behavior_map = {behavior: i for i, behavior in enumerate(behavior_list)}
num_classes = len(behavior_list)

# Salva o mapa para ser usado no treino e inferência
with open(BEHAVIOR_MAP_PATH, 'w') as f:
    json.dump(behavior_map, f, indent=4)
print(f"Mapa de {num_classes} comportamentos salvo em '{BEHAVIOR_MAP_PATH}'")

print("3. Calculando o número de ocorrências positivas para cada classe...")
# Conta as ocorrências de cada comportamento
positive_counts = {b: 0 for b in behavior_list}
for labels in df_y['behavior'].str.split(';'):
    for label in labels:
        if label in positive_counts:
            positive_counts[label] += 1

# Cria um array ordenado de contagens
positive_counts_array = np.array([positive_counts[b] for b in behavior_list])

print("4. Calculando e salvando os pesos (pos_weight)...")
# Evita divisão por zero para classes que nunca aparecem (embora improvável)
positive_counts_array[positive_counts_array == 0] = 1 

# Calcula o peso para cada classe
# peso = (total_de_amostras - amostras_positivas) / amostras_positivas
negative_counts_array = total_samples - positive_counts_array
pos_weight = negative_counts_array / positive_counts_array

# Converte para um tensor numpy e salva
np.save(WEIGHTS_OUTPUT_PATH, pos_weight.astype(np.float32))

print("\n=======================================================")
print(f"✅ Pesos calculados e salvos em '{WEIGHTS_OUTPUT_PATH}'")
print("Pesos (primeiros 10):", pos_weight[:10])
print("=======================================================")