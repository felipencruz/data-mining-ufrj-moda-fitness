import pandas as pd
import numpy as np
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df = pd.read_csv('../outputs/df_vendas_diario.csv', sep=';', parse_dates=['data_venda'])
df = df.sort_values('data_venda').reset_index(drop=True)

# CRIANDO LAGS (Ensinar o modelo a comparar o hoje com o passado)
for lag in [1, 7, 14, 30, 365]:
    df[f'faturamento_lag_{lag}'] = df['faturamento_dia'].shift(lag)

# MÉDIAS MÓVEIS (Suavizar as variações diárias para ver a direção real das vendas)
df['media_movel_7d'] = df['faturamento_dia'].rolling(7).mean()
df['media_movel_30d'] = df['faturamento_dia'].rolling(30).mean()

# VOLATILIDADE DAS VENDAS
df['variacao_7d'] = df['faturamento_dia'].rolling(7).std()
print("Lags e Médias Móveis calculados!")

# LIMPEZA DE DADOS INICIAIS (Removendo o primeiro ano de dados, pois não há "ano passado" para comparar com o lag_365)
df_limpo = df.dropna(subset=['faturamento_lag_365']).reset_index(drop=True)
print(f"Linhas removidas (sem histórico): {len(df) - len(df_limpo)}")

# EXPORTAR BASE
caminho = '../outputs/df_vendas_diario_lags.csv'
df_limpo.to_csv(caminho, index=False, sep=';', encoding='utf-8-sig')

print(f"OK: {caminho}")
print("Memória Temporal Concluída!")