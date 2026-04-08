import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
caminho_input = '../outputs/df_vendas_final.csv'

if not os.path.exists(caminho_input):
    print("Erro: O arquivo 'df_vendas_final.csv' não foi encontrado. Rode o script 05 primeiro!")
else:
    vendas = pd.read_csv(caminho_input, sep=';', parse_dates=['data_venda'])
    
    # AGRUPAMENTO POR DIA (Se 'transacao_id' não existir por erro de rodagem, usamos 'venda_id' para contar)
    coluna_contagem = 'transacao_id' if 'transacao_id' in vendas.columns else 'venda_id'
    vendas_diario = vendas.groupby('data_venda').agg(
        vendas_dia      = (coluna_contagem, 'count'),
        faturamento_dia = ('valor_total', 'sum'),
        unidades_dia    = ('quantidade', 'sum')
    ).reset_index()

    # INTELIGÊNCIA DE TEMPO
    vendas_diario['ano'] = vendas_diario['data_venda'].dt.year
    vendas_diario['mes'] = vendas_diario['data_venda'].dt.month
    vendas_diario['dia_semana'] = vendas_diario['data_venda'].dt.dayofweek
    
    print(f"Resumo diário calculado usando a coluna: {coluna_contagem}")

    # EXPORTAR BASE
    caminho_diario = '../outputs/df_vendas_diario.csv'
    vendas_diario.to_csv(caminho_diario, index=False, sep=';', encoding='utf-8-sig')

    print(f"OK: {caminho_diario}")
    print("Agregação Diária Concluída!")