# Moda Fitness Brasil — Mineração de Dados

Projeto de mineração de dados aplicado à empresa fictícia **Moda Fitness Brasil**, desenvolvido como trabalho final da disciplina de **Data Mining** no MBA em Big Data, Business Intelligence e Business Analytics da **Escola Politécnica da UFRJ**.

## Sobre o Projeto

O pipeline cobre o processo completo de KDD/CRISP-DM: ingestão e limpeza de 251.000 transações de venda, feature engineering, modelagem preditiva (SARIMA, Ridge Regression e LightGBM) e geração de previsão de faturamento para 2026 — com MAPE de 4,9% no conjunto de teste.

## Estrutura

```
inputs/    → arquivos CSV brutos (fVendas, dClientes, dProdutos, fEstoque)
scripts/   → 17 scripts Python numerados sequencialmente
outputs/   → artefatos intermediários e resultados finais
```

Execute os scripts em ordem a partir dos arquivos em `inputs/` para reproduzir todos os resultados.

## Autores

- Carlos Henrique Ramalho P Linhares
- Eralda Ferreira da Silva
- Felipe Nascimento da Cruz
- Flávia Lucena de Araújo
- Rafael Castilho Freire

## Disciplina

Data Mining — MBA em Big Data, Business Intelligence e Business Analytics  
Escola Politécnica da UFRJ  
Professor: Cláudio Latta
