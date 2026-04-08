import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO DAS TABELAS TRATADAS
vendas = pd.read_csv('../outputs/df_vendas_limpo.csv', sep=';')
clientes = pd.read_csv('../outputs/df_clientes_limpo.csv', sep=';')
produtos = pd.read_csv('../outputs/df_produtos_limpo.csv', sep=';')
estoque = pd.read_csv('../outputs/df_estoque_limpo.csv', sep=';')

print("Tabelas carregadas com sucesso!")

# UNIÃO: VENDAS + CLIENTES (Trazer apenas as características do cliente que importam para a análise)
cols_clientes = ['cliente_id', 'genero', 'faixa_etaria', 'uf', 'regiao', 'dias_desde_cadastro']
vendas_final = vendas.merge(clientes[cols_clientes], on='cliente_id', how='left')

# UNIÃO: + PRODUTOS (Adicionar o nome do produto e a categoria à linha da venda)
cols_produtos = ['sku', 'produto', 'categoria']
vendas_final = vendas_final.merge(produtos[cols_produtos], on='sku', how='left')

# UNIÃO: + ESTOQUE (Cruzar com o estoque atual para saber a disponibilidade de cada item vendido)
vendas_final = vendas_final.merge(estoque.rename(columns={'qtd_estoque': 'estoque_atual'}), on='sku', how='left')

# RELATÓRIO DE CONFERÊNCIA
print("\nResumo da Tabela Consolidada:")
print(f"Total de registros: {len(vendas_final)}")
print(f"Número de colunas: {len(vendas_final.columns)}")

# FATURAMENTO POR CATEGORIA (Para validar se os dados fazem sentido)
faturamento_cat = vendas_final.groupby('categoria')['valor_total'].sum().sort_values(ascending=False)
print("\nFaturamento por Categoria:")
print(faturamento_cat.apply(lambda x: f"R$ {x:,.2f}"))

# EXPORTANDO A BASE
print("\nSalvando base analítica final em /outputs!")
caminho_final = '../outputs/df_vendas_final.csv'
vendas_final.to_csv(caminho_final, index=False, sep=';', encoding='utf-8-sig')

print(f"OK: {caminho_final}")
print("Base Analítica Pronta para Análise e Gráficos!")