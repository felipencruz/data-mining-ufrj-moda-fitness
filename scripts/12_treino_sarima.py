import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df = pd.read_csv('../outputs/base_para_sarima.csv', sep=';', parse_dates=['data_venda'])
df = df.set_index('data_venda').sort_index()
df = df.asfreq('D')
df['log_faturamento'] = df['log_faturamento'].fillna(0)

# SEPARAÇÃO DOS ALVOS
y_treino = df.loc['2022-04-01':'2024-12-31', 'log_faturamento']
y_val    = df.loc['2025-01-01':'2025-12-31', 'log_faturamento']
y_teste  = df.loc['2026-01-01':'2026-03-31', 'log_faturamento']

# GRID SEARCH SIMPLIFICADO
print("Buscando os melhores parâmetros (isso pode levar um minuto)...")
melhor_mape = float('inf')
melhor_ordem = None

for p in [0, 1, 2]:
    for q in [0, 1, 2]:
        try:
            modelo = SARIMAX(y_treino, order=(p, 0, q), seasonal_order=(1, 1, 1, 7))
            resultado = modelo.fit(disp=False)
            
            pred = resultado.get_forecast(steps=len(y_val)).predicted_mean.values
            mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(pred))
            
            if mape < melhor_mape:
                melhor_mape = mape
                melhor_ordem = (p, 0, q)
        except:
            continue

print(f"Melhor configuração encontrada: {melhor_ordem} com MAPE de {melhor_mape:.2%}")

# TREINANDO O MODELO FINAL
print("Treinando modelo final...")
modelo_final = SARIMAX(y_treino, order=melhor_ordem, seasonal_order=(1, 1, 1, 7))
res_final = modelo_final.fit(disp=False)

# GERANDO PREVISÕES
passos_totais = len(y_val) + len(y_teste)

pred_val = np.expm1(res_final.get_forecast(steps=len(y_val)).predicted_mean.values)
pred_teste = np.expm1(res_final.get_forecast(steps=passos_totais).predicted_mean.values[-len(y_teste):])

# EXPORTANDO RESULTADOS
print("Salvando previsões do SARIMA!")
pd.DataFrame({
    'data': y_val.index, 
    'real': np.expm1(y_val).values, 
    'previsto': pred_val
}).to_csv('../outputs/pred_sarima_val.csv', index=False, sep=';', encoding='utf-8-sig')

pd.DataFrame({
    'data': y_teste.index, 
    'real': np.expm1(y_teste).values, 
    'previsto': pred_teste
}).to_csv('../outputs/pred_sarima_test.csv', index=False, sep=';', encoding='utf-8-sig')

print("OK: Previsões salvas em /outputs!")
print("Treino SARIMA Concluído!")