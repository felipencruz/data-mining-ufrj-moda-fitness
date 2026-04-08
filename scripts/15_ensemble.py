import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_percentage_error

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAR PREVISÕES
sarima_val = pd.read_csv('../outputs/pred_sarima_val.csv', sep=';')
prophet_val = pd.read_csv('../outputs/pred_prophet_val.csv', sep=';')
lgbm_val = pd.read_csv('../outputs/pred_lgbm_val.csv', sep=';')

sarima_test = pd.read_csv('../outputs/pred_sarima_test.csv', sep=';')
prophet_test = pd.read_csv('../outputs/pred_prophet_test.csv', sep=';')
lgbm_test = pd.read_csv('../outputs/pred_lgbm_test.csv', sep=';')

# Tabelas Consolidadas
val_df = pd.DataFrame({
    'data': lgbm_val['data'], 'real': lgbm_val['real'],
    'sarima': sarima_val['previsto'], 'prophet': prophet_val['previsto'], 'lgbm': lgbm_val['previsto']
})

test_df = pd.DataFrame({
    'data': lgbm_test['data'], 'real': lgbm_test['real'],
    'sarima': sarima_test['previsto'], 'prophet': prophet_test['previsto'], 'lgbm': lgbm_test['previsto']
})

# OTIMIZADOR DE PESOS
print("Buscando a combinação perfeita de pesos (0% a 100%)...")
melhor_mape = float('inf')
melhores_pesos = (0, 0, 0)

# Testa combinações em passos de 5% (0.05)
passos = np.arange(0.0, 1.05, 0.05)

for w_sarima in passos:
    for w_prophet in passos:
        w_lgbm = 1.0 - w_sarima - w_prophet
        
        if -0.001 <= w_lgbm <= 1.001: 
            pred_mistura = (val_df['sarima'] * w_sarima) + (val_df['prophet'] * w_prophet) + (val_df['lgbm'] * w_lgbm)
            mape_atual = mean_absolute_percentage_error(val_df['real'], pred_mistura)

            if mape_atual < melhor_mape:
                melhor_mape = mape_atual
                melhores_pesos = (w_sarima, w_prophet, w_lgbm)

w_sarima, w_prophet, w_lgbm = melhores_pesos

print(f"\n🏆 Pesos Ideais Encontrados na Validação:")
print(f"SARIMA: {w_sarima:.0%} | Prophet: {w_prophet:.0%} | LightGBM: {w_lgbm:.0%}")
print(f"MAPE do Ensemble na Validação: {melhor_mape:.2%}")

# 3. APLICANDO NO TESTE (A Prova Final no Futuro de 2026)
test_df['pred_ensemble'] = (test_df['sarima'] * w_sarima) + (test_df['prophet'] * w_prophet) + (test_df['lgbm'] * w_lgbm)
mape_teste = mean_absolute_percentage_error(test_df['real'], test_df['pred_ensemble'])

print(f"\n🚀 MAPE FINAL NO TESTE (Início de 2026): {mape_teste:.2%}")

# 4. EXPORTANDO RESULTADOS
print("\nSalvando resultados finais!")
test_df.to_csv('../outputs/predicao_final_ensemble.csv', index=False, sep=';', encoding='utf-8-sig')

pd.DataFrame([{'SARIMA': w_sarima, 'Prophet': w_prophet, 'LightGBM': w_lgbm}]).to_csv(
    '../outputs/pesos_ensemble.csv', index=False, sep=';')

print("OK: Previsão Final guardada!")
print("Etapa de Modelagem 100% Concluída!")