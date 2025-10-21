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

# ============================================================================
# ANÁLISE TARGET E PADRÕES TEMPORAIS
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set_palette("husl")

df = pd.read_parquet("hotel_bookings.parquet")
print(f"✅ Dataset carregado: {df.shape[0]:,} linhas\n")

# ----------------------------------------------------------------------------
# Análise da Variável Target (is_canceled)
# ----------------------------------------------------------------------------
print("="*80)
print("ANÁLISE DA VARIÁVEL TARGET: is_canceled")
print("="*80)

cancel_counts = df['is_canceled'].value_counts()
cancel_rate = df['is_canceled'].mean() * 100

print(f"\n📊 DISTRIBUIÇÃO:")
print(f"   Não Cancelado: {cancel_counts[0]:,} ({(cancel_counts[0]/len(df)*100):.2f}%)")
print(f"   Cancelado: {cancel_counts[1]:,} ({(cancel_counts[1]/len(df)*100):.2f}%)")
print(f"\n🎯 Taxa Geral: {cancel_rate:.2f}%")

# Visualização
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(['Não Cancelado', 'Cancelado'], cancel_counts.values, 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[0].set_title('Distribuição de Cancelamentos', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Número de Reservas', fontsize=12)

for i, v in enumerate(cancel_counts.values):
    axes[0].text(i, v + 1000, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')

cancel_by_hotel = df.groupby('hotel')['is_canceled'].mean() * 100
axes[1].bar(cancel_by_hotel.index, cancel_by_hotel.values, 
           color=['#3498db', '#9b59b6'], alpha=0.7, edgecolor='black')
axes[1].set_title('Taxa de Cancelamento por Tipo de Hotel', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Taxa de Cancelamento (%)', fontsize=12)

for i, (hotel, rate) in enumerate(cancel_by_hotel.items()):
    axes[1].text(i, rate + 1, f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n🏨 POR TIPO DE HOTEL:")
for hotel, rate in cancel_by_hotel.items():
    print(f"   {hotel}: {rate:.2f}%")

# ----------------------------------------------------------------------------
# Preparação de Dados Temporais
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ANÁLISE TEMPORAL")
print("="*80)

df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'] + '-' +
    df['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)

print(f"✅ Coluna de data criada!")
print(f"   Período: {df['arrival_date'].min()} a {df['arrival_date'].max()}")

# ----------------------------------------------------------------------------
# Reservas ao Longo do Tempo
# ----------------------------------------------------------------------------
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

monthly_bookings = df.groupby(['arrival_date_year', 'arrival_date_month']).size().unstack(0)
monthly_bookings = monthly_bookings.reindex(month_order)

fig, ax = plt.subplots(figsize=(15, 6))
monthly_bookings.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Reservas por Mês e Ano', fontweight='bold', fontsize=14)
ax.set_xlabel('Mês', fontsize=12)
ax.set_ylabel('Número de Reservas', fontsize=12)
ax.legend(title='Ano')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

monthly_total = monthly_bookings.sum(axis=1)
print(f"\n📊 Maior demanda: {monthly_total.idxmax()} ({monthly_total.max():,})")
print(f"   Menor demanda: {monthly_total.idxmin()} ({monthly_total.min():,})")

# ----------------------------------------------------------------------------
# Taxa de Cancelamento Mensal
# ----------------------------------------------------------------------------
monthly_cancel = df.groupby('arrival_date_month')['is_canceled'].mean() * 100
monthly_cancel = monthly_cancel.reindex(month_order)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(range(len(monthly_cancel)), monthly_cancel.values, 
        marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
ax.set_xticks(range(len(monthly_cancel)))
ax.set_xticklabels(monthly_cancel.index, rotation=45, ha='right')
ax.set_title('Taxa de Cancelamento por Mês', fontweight='bold', fontsize=14)
ax.set_ylabel('Taxa de Cancelamento (%)', fontsize=12)
ax.grid(True, alpha=0.3)

mean_cancel = monthly_cancel.mean()
ax.axhline(mean_cancel, color='blue', linestyle='--', linewidth=2, 
          label=f'Média: {mean_cancel:.2f}%', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()

print(f"\n📊 Maior taxa: {monthly_cancel.idxmax()} ({monthly_cancel.max():.2f}%)")
print(f"   Menor taxa: {monthly_cancel.idxmin()} ({monthly_cancel.min():.2f}%)")

# ----------------------------------------------------------------------------
# Análise de Sazonalidade
# ----------------------------------------------------------------------------
def get_season(month):
    if month in ['December', 'January', 'February']:
        return 'Inverno'
    elif month in ['March', 'April', 'May']:
        return 'Primavera'
    elif month in ['June', 'July', 'August']:
        return 'Verão'
    else:
        return 'Outono'

df['season'] = df['arrival_date_month'].apply(get_season)

season_stats = df.groupby('season').agg({
    'is_canceled': ['count', 'mean']
}).round(3)
season_stats.columns = ['total_reservas', 'taxa_cancelamento']
season_stats['taxa_cancelamento'] = season_stats['taxa_cancelamento'] * 100

print("\n📊 POR ESTAÇÃO:")
display(season_stats.sort_values('total_reservas', ascending=False))

season_order = ['Primavera', 'Verão', 'Outono', 'Inverno']
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

season_counts = df['season'].value_counts().reindex(season_order)
axes[0].bar(season_order, season_counts.values, 
           color=['#2ecc71', '#f39c12', '#e67e22', '#3498db'], alpha=0.7)
axes[0].set_title('Reservas por Estação', fontweight='bold')
axes[0].set_ylabel('Número de Reservas')

season_cancel = df.groupby('season')['is_canceled'].mean() * 100
season_cancel = season_cancel.reindex(season_order)
axes[1].bar(season_order, season_cancel.values, 
           color=['#2ecc71', '#f39c12', '#e67e22', '#3498db'], alpha=0.7)
axes[1].set_title('Taxa de Cancelamento por Estação', fontweight='bold')
axes[1].set_ylabel('Taxa (%)')

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("✅ ANÁLISE TEMPORAL CONCLUÍDA!")
print("="*80)
