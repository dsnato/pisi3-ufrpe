# -*- coding: utf-8 -*-
"""
EDA - Hotel Booking Demand Dataset
Commit 1: Configuração Inicial e Carregamento dos Dados
"""

# ============================================================================
# SETUP INICIAL E CARREGAMENTO DE DADOS
# ============================================================================

print("="*80)
print("CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS DADOS")
print("="*80)

# Importação de Bibliotecas
import os
import zipfile
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Bibliotecas importadas com sucesso!")

# ----------------------------------------------------------------------------
# Download e Descompactação do Dataset
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("PREPARANDO DATASET")
print("="*80)

zip_file_name = "hotel-booking-demand.zip"

if not os.path.exists(zip_file_name):
    print("⚠️  Arquivo não encontrado localmente.")
    print("📥 Para baixar o dataset, execute no terminal:")
    print("   kaggle datasets download -d jessemostipak/hotel-booking-demand")
else:
    print(f"✅ Arquivo '{zip_file_name}' encontrado!")

# Descompactar dataset
if os.path.exists(zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()
    print("✅ Dataset descompactado com sucesso!")
    
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    csv_file_name = csv_files[0] if csv_files else "hotel_bookings.csv"
    print(f"📄 Arquivo CSV: {csv_file_name}")
else:
    csv_file_name = "hotel_bookings.csv"

# ----------------------------------------------------------------------------
# Carregamento e Conversão para Parquet
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("CARREGANDO DATASET")
print("="*80)

parquet_file_name = "hotel_bookings.parquet"

if os.path.exists(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)
    print(f"✅ Dataset carregado do Parquet: {parquet_file_name}")
elif os.path.exists(csv_file_name):
    print(f"📊 Carregando CSV: {csv_file_name}")
    df = pd.read_csv(csv_file_name)
    print(f"✅ Dataset CSV carregado!")
    
    df.to_parquet(parquet_file_name, index=False)
    print(f"💾 Dataset salvo em Parquet: {parquet_file_name}")
else:
    raise FileNotFoundError(f"❌ Arquivo não encontrado: {parquet_file_name} ou {csv_file_name}")

# ----------------------------------------------------------------------------
# Visão Geral do Dataset
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("VISÃO GERAL DO DATASET")
print("="*80)

print(f"\n📊 DIMENSÕES:")
print(f"   Linhas: {df.shape[0]:,}")
print(f"   Colunas: {df.shape[1]}")

print("\n👀 PRIMEIRAS 5 LINHAS:")
display(df.head())

print("\n📋 ÚLTIMAS 5 LINHAS:")
display(df.tail())

print("\n🔍 INFORMAÇÕES GERAIS:")
df.info()

print("\n📝 COLUNAS DO DATASET:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "="*80)
print("✅ CONFIGURAÇÃO CONCLUÍDA!")
print("="*80)

plt.style.use('ggplot')
sns.set_palette("husl")

# Carregar dataset
df = pd.read_parquet("hotel_bookings.parquet")
print(f"✅ Dataset carregado: {df.shape[0]:,} linhas × {df.shape[1]} colunas\n")

# ----------------------------------------------------------------------------
# Análise de Valores Faltantes
# ----------------------------------------------------------------------------
print("="*80)
print("ANÁLISE DE VALORES FALTANTES")
print("="*80)

missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

missing_df = pd.DataFrame({
    'Valores Faltantes': missing_data,
    'Percentual (%)': missing_percent
})

missing_with_nulls = missing_df[missing_df['Valores Faltantes'] > 0]

if len(missing_with_nulls) > 0:
    print("\n📊 COLUNAS COM VALORES FALTANTES:")
    display(missing_with_nulls)
else:
    print("\n✅ Nenhum valor faltante encontrado!")

# Visualização matricial
plt.figure(figsize=(12, 6))
msno.matrix(df, fontsize=10)
plt.title('Matriz de Valores Faltantes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Heatmap de correlação
if len(missing_with_nulls) > 1:
    plt.figure(figsize=(10, 6))
    msno.heatmap(df, fontsize=10)
    plt.title('Correlação entre Valores Faltantes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# Estatísticas Descritivas - Numéricas
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS NUMÉRICAS")
print("="*80)

numeric_stats = df.describe().T
numeric_stats['missing'] = missing_data
numeric_stats['missing_pct'] = missing_percent

print("\n📊 RESUMO ESTATÍSTICO:")
display(numeric_stats)

numeric_stats.to_csv('numeric_statistics.csv')
print("\n💾 Estatísticas salvas em: numeric_statistics.csv")

# ----------------------------------------------------------------------------
# Estatísticas Descritivas - Categóricas
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS CATEGÓRICAS")
print("="*80)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n📋 Total: {len(categorical_cols)} variáveis categóricas\n")

for col in categorical_cols:
    print(f"\n{'='*60}")
    print(f"📌 {col.upper()}")
    print(f"{'='*60}")
    
    value_counts = df[col].value_counts()
    total = len(df)
    
    print(f"   Categorias únicas: {df[col].nunique()}")
    print(f"   Valores faltantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/total*100:.2f}%)")
    print(f"\n   Top 5 valores:")
    
    for idx, (value, count) in enumerate(value_counts.head().items(), 1):
        percentage = (count / total) * 100
        print(f"      {idx}. {value}: {count:,} ({percentage:.2f}%)")

# ----------------------------------------------------------------------------
# Detecção de Outliers
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("DETECÇÃO DE OUTLIERS")
print("="*80)

key_vars = ['lead_time', 'adr', 'stays_in_weekend_nights', 
            'stays_in_week_nights', 'adults', 'children', 'babies']
key_vars = [v for v in key_vars if v in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, col in enumerate(key_vars[:8]):
    axes[idx].boxplot(df[col].dropna())
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].set_ylabel('Valor')
    
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
    
    axes[idx].text(0.5, 0.95, f'Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)',
                   transform=axes[idx].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✅ ANÁLISE DE QUALIDADE CONCLUÍDA!")
print("="*80)