import pandas as pd
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO DOS DADOS LIMPOS
vendas = pd.read_csv('../outputs/df_vendas_limpo.csv', sep=';')

# ANALISE DE DUPLICATAS
campos_chave = ['venda_id', 'data_venda', 'cliente_id', 'sku']
total_duplicadas = vendas.duplicated(subset=campos_chave).sum()

print(f"Registros no início: {len(vendas)}")
print(f"Duplicatas encontradas: {total_duplicadas}")

# REMOÇÃO DE DUPLICATAS
if total_duplicadas > 0:
    vendas = vendas.drop_duplicates(subset=campos_chave, keep='first')
    print(f"Limpeza concluída! Registros restantes: {len(vendas)}")
else:
    print("Nenhuma duplicata encontrada. Seguindo processo...")

# CRIAÇÃO DE ID ÚNICO
vendas = vendas.reset_index(drop=True)
vendas.insert(0, 'transacao_id', range(1, len(vendas) + 1))

# EXPORTANDO O ARQUIVO ATUALIZADO
print("\nAtualizando arquivo de vendas em /outputs...")
caminho_final = '../outputs/df_vendas_limpo.csv'
vendas.to_csv(caminho_final, index=False, sep=';', encoding='utf-8-sig')

print(f"  OK: {caminho_final}")
print("\nProcesso de Deduplicação Concluído!")