# Processamento de Dados Otimizado para CatBoost

Este documento explica as otimizações feitas no pipeline de processamento de dados para uso eficiente com métodos de árvore como CatBoost, baseado nas melhores práticas do desafio MABe do Kaggle.

## 📋 Principais Mudanças

### 1. **Variáveis Categóricas Mantidas como Categóricas**
- **Antes**: One-Hot Encoding (OHE) era aplicado, criando muitas colunas esparsas
- **Agora**: Variáveis categóricas são mantidas como tipo `category` e passadas diretamente ao CatBoost
- **Benefício**: CatBoost processa categóricas de forma nativa e eficiente, sem necessidade de OHE

### 2. **Remoção de Normalização Desnecessária**
- **Antes**: Features numéricas eram normalizadas (Z-score)
- **Agora**: Features numéricas são mantidas em escala original
- **Benefício**: Métodos de árvore como CatBoost não precisam de normalização e podem se beneficiar da escala original

### 3. **Features Temporais Adicionadas**
- **Lags temporais**: Valores anteriores (1, 2, 3, 5, 10 frames atrás)
- **Rolling statistics**: Média, desvio padrão, máximo e mínimo em janelas móveis (3, 5, 10, 20 frames)
- **Diferenças temporais**: Derivadas de primeira ordem
- **Benefício**: Captura padrões temporais essenciais para comportamento de ratos

### 4. **Features de Interação**
- Velocidades relativas entre ratos
- Proximidade ponderada por velocidade
- **Benefício**: Melhora a detecção de comportamentos sociais

### 5. **Tratamento de Valores Ausentes**
- CatBoost lida nativamente com NaNs
- Imputação mínima apenas para economizar espaço (opcional)
- **Benefício**: Menos pré-processamento, mais robustez

## 🚀 Como Usar

### Passo 1: Processar Dados com Pipeline Otimizado

```bash
python process_data_catboost.py
```

Este script:
- Processa arquivos Parquet raw
- Adiciona features temporais
- Mantém variáveis categóricas como categóricas
- Salva em `MABe-mouse-behavior-detection/feature_engineered_data_catboost/`

### Passo 2: Consolidar Dados

```bash
python consolidate_data_catboost.py
```

Este script:
- Consolida todos os arquivos processados
- Separa features numéricas, categóricas e labels
- Salva:
  - `consolidated_X_catboost.npy` - Features numéricas
  - `consolidated_X_catboost_categorical.parquet` - Variáveis categóricas
  - `consolidated_Y_catboost.csv` - Labels
  - `categorical_info_catboost.pkl` - Metadados sobre categóricas

### Passo 3: Treinar Modelo CatBoost

```bash
python train_catboost_example.py
```

Este script demonstra:
- Como carregar os dados consolidados
- Como preparar variáveis categóricas para CatBoost
- Como treinar um modelo multi-label
- Como avaliar e salvar o modelo

## 📊 Estrutura de Dados

### Features Numéricas
- Coordenadas normalizadas (cm)
- Velocidades e acelerações
- Distâncias sociais
- Ângulos corporais
- **Features temporais** (lags, rolling stats)
- **Features de interação**

### Variáveis Categóricas
- `arena_type`, `arena_shape`
- `mouse1_sex`, `mouse2_sex`, `mouse3_sex`, `mouse4_sex`
- `mouse1_strain`, `mouse2_strain`, etc.
- `mouse1_color`, `mouse2_color`, etc.
- `mouse1_condition`, `mouse2_condition`, etc.
- `lab_id`, `tracking_method`

### Labels
- Multi-label: cada frame pode ter múltiplos comportamentos
- Formato: string com labels separados por `;`

## 🔧 Configurações Avançadas

### Ajustar Janelas Temporais

No arquivo `process_data_catboost.py`, você pode ajustar:

```python
TEMPORAL_WINDOWS = [3, 5, 10, 20]  # Janelas para rolling statistics
LAG_FEATURES = [1, 2, 3, 5, 10]    # Lags temporais
```

### Ajustar Hiperparâmetros do CatBoost

No arquivo `train_catboost_example.py`:

```python
model = CatBoostClassifier(
    iterations=500,        # Aumente para melhor performance
    learning_rate=0.1,    # Diminua para treinamento mais estável
    depth=6,              # Profundidade das árvores
    loss_function='Logloss',
    eval_metric='AUC',
    cat_features=cat_feature_indices,  # IMPORTANTE: especifica categóricas
    task_type='CPU',      # Mude para 'GPU' se disponível
)
```

## 💡 Dicas de Otimização

1. **Use GPU**: Se disponível, mude `task_type='GPU'` para acelerar o treinamento
2. **Early Stopping**: Já configurado para evitar overfitting
3. **Validação Cruzada**: Considere adicionar k-fold para avaliação mais robusta
4. **Feature Selection**: Após o primeiro treinamento, analise feature importance
5. **Threshold Tuning**: Ajuste o threshold (padrão 0.5) para predições binárias

## 📈 Comparação com Pipeline Original

| Aspecto | Pipeline Original | Pipeline CatBoost |
|---------|------------------|-------------------|
| Variáveis Categóricas | OHE (muitas colunas) | Categóricas nativas |
| Normalização | Z-score | Sem normalização |
| Features Temporais | Limitadas | Extensivas (lags, rolling) |
| Tratamento de NaNs | Imputação | Nativo do CatBoost |
| Tamanho dos Dados | Maior (OHE) | Menor (categóricas) |
| Performance | Boa para redes neurais | Otimizada para árvores |

## 🐛 Troubleshooting

### Erro: "Memory Error"
- Reduza `TEMPORAL_WINDOWS` e `LAG_FEATURES`
- Processe arquivos em lotes menores
- Use `consolidate_data_catboost.py` com processamento incremental

### Erro: "Categorical features not found"
- Verifique se `cat_feature_indices` está correto
- Certifique-se de que as categóricas foram adicionadas ao array X

### Performance Lenta
- Use GPU (`task_type='GPU'`)
- Reduza `iterations` durante testes
- Use amostragem para desenvolvimento

## 📚 Referências

- [Documentação CatBoost](https://catboost.ai/)
- [Kaggle MABe Challenge](https://www.kaggle.com/competitions/mabe-2024)
- [Best Practices for Tabular Data](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)

## ✅ Checklist de Uso

- [ ] Dados raw processados com `process_data_catboost.py`
- [ ] Dados consolidados com `consolidate_data_catboost.py`
- [ ] Modelo treinado com `train_catboost_example.py`
- [ ] Hiperparâmetros ajustados
- [ ] Performance avaliada
- [ ] Modelo salvo para inferência

