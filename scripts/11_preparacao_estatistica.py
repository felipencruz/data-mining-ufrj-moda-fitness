import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df = pd.read_csv('../outputs/df_vendas_diario.csv', sep=';', parse_dates=['data_venda'])
df = df.set_index('data_venda').sort_index()

# TRANSFORMAÇÃO DE LOG
df['log_faturamento'] = np.log1p(df['faturamento_dia'])

# TESTE DE ESTABILIDADE
res = adfuller(df['faturamento_dia'])
print(f"Teste de estabilidade (p-value): {res[1]:.4f}")

if res[1] < 0.05:
    print("A série é estável e pronta para o modelo!")
else:
    print("A série precisará de ajustes automáticos pelo SARIMA!")

# EXPORTAR BASE PARA SARIMA
caminho = '../outputs/base_para_sarima.csv'
df.to_csv(caminho, sep=';', encoding='utf-8-sig')

print(f"OK: {caminho}")
print("Ajuste Matemático Concluído!")