import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
vendas = pd.read_csv('../outputs/df_vendas_final.csv', sep=';', parse_dates=['data_venda'])

# COLUNA DE MÊS/ANO
vendas['mes_referencia'] = vendas['data_venda'].dt.to_period('M')

# AGRUPAMENTO POR MÊS
col_id = 'transacao_id' if 'transacao_id' in vendas.columns else 'venda_id'
vendas_mensal = vendas.groupby('mes_referencia').agg(
    faturamento_mes = ('valor_total', 'sum'),
    total_vendas    = (col_id, 'count'),
    clientes_unicos = ('cliente_id', 'nunique')
).reset_index()

# CÁLCULO DE CRESCIMENTO (Variação com o mês anterior)
vendas_mensal['crescimento_pct'] = (vendas_mensal['faturamento_mes'].pct_change() * 100).round(2)
print(f"Métricas mensais calculadas usando: {col_id}")

# EXPORTAR BASE
caminho_mensal = '../outputs/df_vendas_mensal.csv'
vendas_mensal.to_csv(caminho_mensal, index=False, sep=';', encoding='utf-8-sig')

print(f"OK: {caminho_mensal}")
print("Agregação Mensal Concluída!")