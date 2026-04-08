import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('../outputs', exist_ok=True)

# CARREGAMENTO DOS DADOS
tipos_vendas = {
    'venda_id': 'int32', 'cliente_id': 'int32', 'sku': 'category',
    'quantidade': 'int8', 'canal_venda': 'category', 'forma_pagamento': 'category',
    'preco_base': 'float32', 'preco_unitario': 'float32', 'valor_total': 'float32'
}

# ARQUIVOS ORIGINAIS
clientes = pd.read_csv('../inputs/dClientes.csv', sep=';', parse_dates=['data_cadastro'])
produtos = pd.read_csv('../inputs/dProdutos.csv', sep=';', dtype={'sku': 'category'})
estoque  = pd.read_csv('../inputs/fEstoque.csv', sep=';', dtype={'sku': 'category'})
vendas   = pd.read_csv('../inputs/fVendas.csv', sep=';', dtype=tipos_vendas, parse_dates=['data_venda'])

# TRATAMENTO DE NULOS

# GENÊRO PREENCHIDO COM A MODA DA UF
print("Tratando nulos...")
clientes['genero'] = clientes.groupby('uf')['genero'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Não Informado"))

# VENDAS PREENCHIDA COM A MODA GLOBAL
for col in ['canal_venda', 'forma_pagamento']:
    vendas[col] = vendas[col].fillna(vendas[col].mode()[0])

# VENDAS CALCULADAS PELO PREÇO x QUANTIDADE
vendas['preco_unitario'] = vendas['preco_unitario'].fillna(vendas['preco_base'])
vendas['valor_total']    = vendas['valor_total'].fillna(vendas['quantidade'] * vendas['preco_unitario'])

# EXPORTANDO EM CSV
print("\nSalvando arquivos limpos em /outputs...")
tabelas = {
    'clientes': clientes, 
    'produtos': produtos, 
    'estoque': estoque, 
    'vendas': vendas
}

for nome, df in tabelas.items():
    caminho = f'../outputs/df_{nome}_limpo.csv'
    df.to_csv(caminho, index=False, sep=';', encoding='utf-8-sig')
    print(f"  OK: {caminho}")

# VALIDAÇÃO DE INTEGRIDADE
print("\nIniciando Validação de Integridade!")

# CONFERIR ÓRFÃOS (Verificar se há clientes ou produtos nas vendas que não estão cadastrados)
clientes_venda = vendas['cliente_id'].unique()
clientes_base  = clientes['cliente_id'].unique()
orfaos_clientes = [c for c in clientes_venda if c not in clientes_base]

skus_venda = vendas['sku'].unique()
skus_base  = produtos['sku'].unique()
orfaos_produtos = [s for s in skus_venda if s not in skus_base]

print(f"Clientes órfãos encontrados: {len(orfaos_clientes)}")
print(f"Produtos órfãos encontrados: {len(orfaos_produtos)}")

# VALIDAÇÃO DE REGRAS DE NEGÓCIO
data_min = vendas['data_venda'].min().date()
data_max = vendas['data_venda'].max().date()
vendas_negativas = (vendas['valor_total'] < 0).sum()

print(f"Período das vendas: {data_min} até {data_max}")
print(f"Vendas com valor negativo: {vendas_negativas}")

# RELATÓRIO FINAL DE QUALIDADE (Resumo rápido para conferir se está tudo OK)
print("\nGerando relatório de qualidade!")

relatorio = pd.DataFrame({
    'Métrica': ['Total Linhas Vendas', 'Clientes Órfãos', 'Produtos Órfãos', 'Valores Negativos', 'Nulos Restantes'],
    'Resultado': [
        len(vendas), 
        len(orfaos_clientes), 
        len(orfaos_produtos), 
        vendas_negativas,
        vendas.isnull().sum().sum()
    ]
})

# EXPORTANDO RELATORIO
caminho_relatorio = '../outputs/relatorio_qualidade.csv'
relatorio.to_csv(caminho_relatorio, index=False, sep=';', encoding='utf-8-sig')

print(f"  OK: {caminho_relatorio}")
print("\nEtapa de Validação Concluída!")
print("Dados prontos para análise!")