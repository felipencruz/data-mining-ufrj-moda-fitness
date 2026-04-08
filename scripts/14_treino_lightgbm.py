import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
treino = pd.read_csv('../outputs/treino.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')
validacao = pd.read_csv('../outputs/validacao.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')
teste = pd.read_csv('../outputs/teste.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')

# SEPARAÇÃO DE FEATURES (X) E ALVO (y)
colunas_remover = ['faturamento_dia']

X_treino = treino.drop(columns=colunas_remover).select_dtypes(include=[np.number])
X_val = validacao.drop(columns=colunas_remover).select_dtypes(include=[np.number])
X_teste = teste.drop(columns=colunas_remover).select_dtypes(include=[np.number])

y_treino = np.log1p(treino['faturamento_dia'])
y_val = np.log1p(validacao['faturamento_dia'])
y_teste = np.log1p(teste['faturamento_dia'])

# TREINAMENTO DO MODELO
print("Treinando o batalhão de Árvores de Decisão...")
modelo = LGBMRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=6, 
    random_state=42,
    verbosity=-1
)

modelo.fit(X_treino, y_treino)

# RANKING DE IMPORTÂNCIA
importancia = pd.DataFrame({
    'Feature': X_treino.columns,
    'Importancia': modelo.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\nTop 5 fatores que mais influenciam as vendas:")
print(importancia.head(5).to_string(index=False))

# PREVISÕES E AVALIAÇÃO
pred_val = modelo.predict(X_val)
pred_teste = modelo.predict(X_teste)

mape_val = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(pred_val))

print(f"\nMAPE na Validação: {mape_val:.2%}")

# EXPORTANDO RESULTADOS
print("\nSalvando previsões do LightGBM!")
pd.DataFrame({
    'data': validacao.index, 
    'real': np.expm1(y_val).values, 
    'previsto': np.expm1(pred_val)
}).to_csv('../outputs/pred_lgbm_val.csv', index=False, sep=';', encoding='utf-8-sig')

pd.DataFrame({
    'data': teste.index, 
    'real': np.expm1(y_teste).values, 
    'previsto': np.expm1(pred_teste)
}).to_csv('../outputs/pred_lgbm_test.csv', index=False, sep=';', encoding='utf-8-sig')

importancia.to_csv('../outputs/lgbm_importancia.csv', index=False, sep=';', encoding='utf-8-sig')

print("OK: Previsões salvas em /outputs!")
print("Treino LightGBM Concluído!")