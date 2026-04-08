import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
clientes = pd.read_csv('../outputs/df_clientes_limpo.csv', sep=';', parse_dates=['data_cadastro'])

# CRIANDO FAIXA ETÁRIA
# Agrupar as idades em blocos para facilitar a análise de perfil
bins = [0, 25, 35, 45, 55, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '55+']

clientes['faixa_etaria'] = pd.cut(clientes['idade'], bins=bins, labels=labels)
print("Faixa etária criada!")

# CALCULANDO DIAS DE CADASTRO (há quanto tempo o cliente está na nossa base até a data de hoje)
data_referencia = pd.Timestamp('2026-03-30')
clientes['dias_desde_cadastro'] = (data_referencia - clientes['data_cadastro']).dt.days
print("Tempo de cadastro calculado!")

# MAPEANDO REGIÕES GEOGRÁFICAS (Transformamos os Estados (UF) em Regiões para análises macro)
regiao_map = {
    'SP': 'Sudeste', 'RJ': 'Sudeste', 'MG': 'Sudeste', 'ES': 'Sudeste',
    'PR': 'Sul', 'SC': 'Sul', 'RS': 'Sul'
}
clientes['regiao'] = clientes['uf'].map(regiao_map)
print("Regiões mapeadas!")

# EXPORTANDO O ARQUIVO ATUALIZADO
print("\nAtualizando arquivo de clientes em /outputs!")
caminho_final = '../outputs/df_clientes_limpo.csv'
clientes.to_csv(caminho_final, index=False, sep=';', encoding='utf-8-sig')

print(f"  OK: {caminho_final}")
print("Processo de Feature Engineering Concluído!")