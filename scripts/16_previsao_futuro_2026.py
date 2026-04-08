import pandas as pd
import numpy as np
import os
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAR BASES
df_diario = pd.read_csv('../outputs/df_vendas_diario.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')
df_completo = pd.read_csv('../outputs/df_dataset_previsao.csv', sep=';', parse_dates=['data_venda']).set_index('data_venda')

pesos = pd.read_csv('../outputs/pesos_ensemble.csv', sep=';').iloc[0]
w_prophet, w_lgbm = pesos['Prophet'], pesos['LightGBM']

print(f"Usando Pesos do Otimizador -> LightGBM: {w_lgbm:.0%} | Prophet: {w_prophet:.0%}")

# TREINAR MODELOS COM DOS DADOS HISTÓRICOS
X_total_lgbm = df_completo.drop(columns=['faturamento_dia']).select_dtypes(include=[np.number])
y_total = np.log1p(df_completo['faturamento_dia'])

print("Treinando modelos finais com toda a base de dados...")
modelo_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbosity=-1)
modelo_lgbm.fit(X_total_lgbm, y_total)

# Preparar Prophet
features_prophet = [c for c in X_total_lgbm.columns if c.startswith('onda_') or c.startswith('is_') or c.startswith('dia_')] + ['tendencia']
data_zero = df_completo.index.min()
df_completo['tendencia'] = (df_completo.index - data_zero).days
X_total_prophet = df_completo[features_prophet]

scaler = StandardScaler()
X_prophet_scaled = scaler.fit_transform(X_total_prophet)
modelo_prophet = Ridge(alpha=100.0)
modelo_prophet.fit(X_prophet_scaled, y_total)

# CRIAR O CALENDÁRIO DO FUTURO
data_inicio_futuro = pd.to_datetime('2026-04-01')
data_fim_futuro = pd.to_datetime('2026-12-31')
datas_futuras = pd.date_range(data_inicio_futuro, data_fim_futuro, freq='D')

df_futuro = pd.DataFrame(index=datas_futuras)
df_futuro['mes'] = df_futuro.index.month
df_futuro['dia_semana'] = df_futuro.index.dayofweek
df_futuro['tendencia'] = (df_futuro.index - data_zero).days

# Ondas Fourier para o futuro
t = np.arange(len(df_completo), len(df_completo) + len(df_futuro))
T = 365
for k in range(1, 3):
    df_futuro[f'onda_sen_{k}'] = np.sin(2 * np.pi * k * t / T)
    df_futuro[f'onda_cos_{k}'] = np.cos(2 * np.pi * k * t / T)

# Feriados do Futuro
df_futuro['is_black_friday'] = df_futuro['mes'].isin([10, 11]).astype(int)
df_futuro['is_verao'] = df_futuro['mes'].isin([12, 1, 2, 3]).astype(int)
df_futuro['is_ferias'] = df_futuro['mes'].isin([7, 8]).astype(int)

# Dias da Semana
dias_dummies = pd.get_dummies(df_futuro['dia_semana'], prefix='dia').astype(int)
for i in range(7):
    if f'dia_{i}' not in dias_dummies.columns:
        dias_dummies[f'dia_{i}'] = 0
df_futuro = pd.concat([df_futuro, dias_dummies], axis=1)

# Previsão Prophet para o Futuro
X_futuro_prophet = df_futuro[features_prophet]
pred_prophet_futuro = np.expm1(modelo_prophet.predict(scaler.transform(X_futuro_prophet)))

# PREVISÃO RECURSIVA DO LIGHTGBM
print("Gerando o futuro dia após dia com o LightGBM (Previsão Recursiva)...")

faturamento_simulado = list(df_completo['faturamento_dia'].values)
previsoes_lgbm = []

colunas_lgbm = X_total_lgbm.columns

for data_atual in datas_futuras:
    linha_hoje = pd.DataFrame(index=[data_atual], columns=colunas_lgbm)
    
    for lag in [1, 7, 14, 30, 365]:
        if f'faturamento_lag_{lag}' in linha_hoje:
            linha_hoje.loc[data_atual, f'faturamento_lag_{lag}'] = faturamento_simulado[-lag]
            
    if 'media_movel_7d' in linha_hoje:
        linha_hoje.loc[data_atual, 'media_movel_7d'] = np.mean(faturamento_simulado[-7:])
    if 'media_movel_30d' in linha_hoje:
        linha_hoje.loc[data_atual, 'media_movel_30d'] = np.mean(faturamento_simulado[-30:])
    if 'variacao_7d' in linha_hoje:
        linha_hoje.loc[data_atual, 'variacao_7d'] = np.std(faturamento_simulado[-7:])
        
    for col in [c for c in colunas_lgbm if c.startswith('onda_') or c.startswith('is_') or c.startswith('dia_')]:
        if col in df_futuro.columns:
            linha_hoje.loc[data_atual, col] = df_futuro.loc[data_atual, col]
            
    linha_hoje = linha_hoje.fillna(X_total_lgbm.mean()).astype(float)
    
    pred_hoje = np.expm1(modelo_lgbm.predict(linha_hoje))[0]
    previsoes_lgbm.append(pred_hoje)
    
    faturamento_simulado.append(pred_hoje)

# O ENSEMBLE DO FUTURO
print("Combinando tudo na Super Previsão...")
df_futuro['pred_lgbm'] = previsoes_lgbm
df_futuro['pred_prophet'] = pred_prophet_futuro
df_futuro['faturamento_previsto'] = (df_futuro['pred_lgbm'] * w_lgbm) + (df_futuro['pred_prophet'] * w_prophet)

# EXPORTANDO O RESULTADO FINANCEIRO
print("\nSalvando a Tabela do Futuro (Abril a Dezembro 2026)!")
caminho = '../outputs/previsao_financeira_2026.csv'
df_futuro[['faturamento_previsto']].round(2).to_csv(caminho, sep=';')

# RELATÓRIO
faturamento_restante = df_futuro['faturamento_previsto'].sum()
print(f"\n==============================================")
print(f"💰 PROJEÇÃO FINAL DE RECEITA (Abr-Dez 2026): R$ {faturamento_restante:,.2f}")
print(f"==============================================")
print(f"OK: {caminho}")
print("Concluído com Sucesso!")