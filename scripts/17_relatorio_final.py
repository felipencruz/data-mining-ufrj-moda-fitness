import pandas as pd
import numpy as np
import os

# CAMINHOS
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# CARREGAMENTO
df_diario = pd.read_csv('../outputs/df_vendas_diario.csv', sep=';', parse_dates=['data_venda'])
pesos = pd.read_csv('../outputs/pesos_ensemble.csv', sep=';').iloc[0]

df_futuro = pd.read_csv('../outputs/previsao_financeira_2026.csv', sep=';')
df_futuro.rename(columns={df_futuro.columns[0]: 'data_venda'}, inplace=True)
df_futuro['data_venda'] = pd.to_datetime(df_futuro['data_venda'])

vendas_2025 = df_diario[df_diario['data_venda'].dt.year == 2025]['faturamento_dia'].sum()
vendas_2026_real = df_diario[(df_diario['data_venda'].dt.year == 2026) & (df_diario['data_venda'].dt.month <= 3)]['faturamento_dia'].sum()
vendas_2026_previsto = df_futuro['faturamento_previsto'].sum()
total_2026 = vendas_2026_real + vendas_2026_previsto
crescimento = ((total_2026 - vendas_2025) / vendas_2025) * 100

# RELATORIO
print("\n" + "="*60)
print(" 📊 RELATÓRIO EXECUTIVO - UFRJ MODA FITNESS (2025 vs 2026)")
print("="*60)
print(f"💰 Faturamento Real (2025):       R$ {vendas_2025:>15,.2f}")
print(f"⏳ Faturamento Real (Jan-Mar/26): R$ {vendas_2026_real:>15,.2f}")
print(f"🔮 Faturamento Previsão (IA):     R$ {vendas_2026_previsto:>15,.2f}")
print("-" * 60)
print(f"🏆 TOTAL PROJETADO (2026):        R$ {total_2026:>15,.2f}")
print(f"🚀 CRESCIMENTO ESPERADO:          {crescimento:>14.2f}%")
print("="*60)
print("🧠 ARQUITETURA DA INTELIGÊNCIA ARTIFICIAL (Erro: ~4.9%)")
print(f"   - LightGBM (Memória de Vendas):     {pesos['LightGBM']:.0%}")
print(f"   - Prophet-Style (Feriados/Clima):   {pesos['Prophet']:.0%}")
print(f"   - SARIMA (Estatística Pura):        {pesos['SARIMA']:.0%}")
print("="*60)

# EXPORTAR RESUMO
resumo = pd.DataFrame([
    {'Metrica': 'Faturamento Real 2025', 'Valor (R$)': round(vendas_2025, 2)},
    {'Metrica': 'Faturamento Real 2026 (Jan-Mar)', 'Valor (R$)': round(vendas_2026_real, 2)},
    {'Metrica': 'Faturamento Previsto 2026 (Abr-Dez)', 'Valor (R$)': round(vendas_2026_previsto, 2)},
    {'Metrica': 'Faturamento Total Projetado 2026', 'Valor (R$)': round(total_2026, 2)},
    {'Metrica': 'Crescimento Anual (%)', 'Valor (R$)': round(crescimento, 2)}
])

caminho = '../outputs/relatorio_executivo_final.csv'
resumo.to_csv(caminho, sep=';', index=False, encoding='utf-8-sig')

print(f"\n✅ OK: Resumo guardado em {caminho}")
print("PIPELINE FINALIZADO!")