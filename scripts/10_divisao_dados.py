import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df = pd.read_csv('../outputs/df_dataset_previsao.csv', sep=';', parse_dates=['data_venda'])
df = df.set_index('data_venda').sort_index()

# DIVISÃO TEMPORAL (Validação: O ano de 2025 e Teste: O início de 2026)
treino = df.loc['2022-04-01':'2024-12-31']
validacao = df.loc['2025-01-01':'2025-12-31']
teste = df.loc['2026-01-01':'2026-03-31']

print(f"Treino: {len(treino)} dias | Validação: {len(validacao)} dias | Teste: {len(teste)} dias")

# EXPORTAR
treino.to_csv('../outputs/treino.csv', sep=';', encoding='utf-8-sig')
validacao.to_csv('../outputs/validacao.csv', sep=';', encoding='utf-8-sig')
teste.to_csv('../outputs/teste.csv', sep=';', encoding='utf-8-sig')

print("OK: Ficheiros de treino, validação e teste salvos!")