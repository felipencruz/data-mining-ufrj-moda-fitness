import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO DOS DADOS (Vendas e Clientes para o cruzamento)
vendas = pd.read_csv('../outputs/df_vendas_limpo.csv', sep=';', parse_dates=['data_venda'])
clientes = pd.read_csv('../outputs/df_clientes_limpo.csv', sep=';', parse_dates=['data_cadastro'])

# COMPONENTES TEMPORAIS (Quebrar a data em partes menores para analisar sazonalidade)
vendas['ano'] = vendas['data_venda'].dt.year
vendas['mes'] = vendas['data_venda'].dt.month
vendas['dia_semana'] = vendas['data_venda'].dt.dayofweek
vendas['nome_dia'] = vendas['data_venda'].dt.day_name()
vendas['eh_fim_semana'] = (vendas['dia_semana'] >= 5).astype(int)
print("Componentes de data criados!")

# IDENTIFICAÇÃO DE CLIENTE NOVO (Cruzar com a base de clientes para ver se a compra foi feita até 30 dias após o cadastro)
vendas = vendas.merge(clientes[['cliente_id', 'data_cadastro']], on='cliente_id', how='left')
vendas['cliente_novo'] = ((vendas['data_venda'] - vendas['data_cadastro']).dt.days <= 30).astype(int)
vendas = vendas.drop(columns=['data_cadastro'])
print("Status de cliente novo identificado!")

# CLASSIFICAÇÃO DE SAZONALIDADE (Criar categorias baseadas no mês para identificar picos de venda)
def classificar_sazonalidade(mes):
    if mes in [1, 12]: return 'Festas'
    if mes in [10, 11]: return 'Black Friday'
    if mes in [7, 8]: return 'Férias'
    if mes in [3, 4]: return 'Pós-Carnaval'
    return 'Regular'

vendas['sazonalidade'] = vendas['mes'].apply(classificar_sazonalidade)
print("Sazonalidade mapeada!")

# MÉTRICAS DE PREÇO E DESCONTO (Calcular a agressividade das promoções e o preço real por item)
vendas['desconto_pct'] = ((vendas['preco_base'] - vendas['preco_unitario']) / vendas['preco_base'] * 100).round(2)
vendas['tem_desconto'] = (vendas['desconto_pct'] > 0.1).astype(int)
vendas['preco_item'] = (vendas['valor_total'] / vendas['quantidade']).round(2)
print("Métricas de desconto calculadas!")

# EXPORTANDO O ARQUIVO ATUALIZADO
print("\nAtualizando arquivo de vendas em /outputs!")
caminho_final = '../outputs/df_vendas_limpo.csv'
vendas.to_csv(caminho_final, index=False, sep=';', encoding='utf-8-sig')

print(f"  OK: {caminho_final}")
print("Processo de Feature Engineering em Vendas Concluído!")