import pandas as pd
import numpy as np
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df = pd.read_csv('../outputs/df_vendas_diario_lags.csv', sep=';', parse_dates=['data_venda'])

# FOURIER (Transformar o tempo em "ondas" matemáticas para o modelo entender ciclos)
t = np.arange(len(df))
T = 365
for k in range(1, 3): # Criamos 2 pares de ondas principais
    df[f'onda_sen_{k}'] = np.sin(2 * np.pi * k * t / T)
    df[f'onda_cos_{k}'] = np.cos(2 * np.pi * k * t / T)
print("Ondas de sazonalidade criadas!")

# DUMMIES DE EVENTOS (Identificar épocas críticas para o comércio fitness)
df['is_black_friday'] = df['mes'].isin([10, 11]).astype(int)
df['is_verao'] = df['mes'].isin([12, 1, 2, 3]).astype(int)
df['is_ferias'] = df['mes'].isin([7, 8]).astype(int)

# DIAS DA SEMANA (Criar colunas separadas para que o modelo aprenda o peso de cada dia)
dias_dummies = pd.get_dummies(df['dia_semana'], prefix='dia').astype(int)
df = pd.concat([df, dias_dummies], axis=1)
print("Eventos e dias da semana mapeados!")

# EXPORTAR BASE
caminho_final = '../outputs/df_dataset_previsao.csv'
df.to_csv(caminho_final, index=False, sep=';', encoding='utf-8-sig')

print(f"OK: {caminho_final}")
print("Dataset Final para IA Concluído!")