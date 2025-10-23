import pandas as pd

df = pd.read_csv('test_predictions_final_ANALYSIS.csv')
print(df.head())
print(df['predicted_behavior_argmax'].value_counts())