import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
treino = pd.read_csv('../outputs/treino.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')
validacao = pd.read_csv('../outputs/validacao.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')
teste = pd.read_csv('../outputs/teste.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')

# CRIANDO COLUNA DE TENDÊNCIA
data_zero = treino.index.min()
for df in [treino, validacao, teste]:
    df['tendencia'] = (df.index - data_zero).days

# SELECIONANDO FEATURES
features = [c for c in treino.columns if 'onda_' in c or 'is_' in c or 'dia_' in c] + ['tendencia']

X_treino = treino[features]
X_val = validacao[features]
X_teste = teste[features]

# ALVO
y_treino = np.log1p(treino['faturamento_dia'])
y_val = np.log1p(validacao['faturamento_dia'])
y_teste = np.log1p(teste['faturamento_dia'])

#PADRONIZAÇÃO
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_val_scaled = scaler.transform(X_val)
X_teste_scaled = scaler.transform(X_teste)

# TREINAMENTO
print("Buscando os melhores parâmetros...")
melhor_mape = float('inf')
melhor_modelo = None

for alpha in [0.1, 1.0, 10.0, 100.0, 500.0]:
    modelo = Ridge(alpha=alpha)
    modelo.fit(X_treino_scaled, y_treino)
    
    pred = modelo.predict(X_val_scaled)
    mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(pred))
    
    if mape < melhor_mape:
        melhor_mape = mape
        melhor_modelo = modelo

print(f"Melhor modelo encontrado com MAPE de {melhor_mape:.2%} na Validação!")

# PREVISÕES FINAIS
pred_val_final = np.expm1(melhor_modelo.predict(X_val_scaled))
pred_teste_final = np.expm1(melhor_modelo.predict(X_teste_scaled))

# EXPORTANDO RESULTADOS
print("Salvando previsões do Prophet-Style!")
pd.DataFrame({
    'data': validacao.index, 
    'real': np.expm1(y_val).values, 
    'previsto': pred_val_final
}).to_csv('../outputs/pred_prophet_val.csv', index=False, sep=';', encoding='utf-8-sig')

pd.DataFrame({
    'data': teste.index, 
    'real': np.expm1(y_teste).values, 
    'previsto': pred_teste_final
}).to_csv('../outputs/pred_prophet_test.csv', index=False, sep=';', encoding='utf-8-sig')

print("OK: Previsões salvas em /outputs!")
print("Treino Prophet-Style Concluído!")