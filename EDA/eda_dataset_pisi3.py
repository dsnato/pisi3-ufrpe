# -*- coding: utf-8 -*-
"""
EDA - Hotel Booking Demand Dataset
Análise Exploratória de Dados Completa
"""

# ============================================================================
# CONFIGURAÇÃO INICIAL E IMPORTAÇÕES
# ============================================================================

print("="*80)
print("🔍 ANÁLISE EXPLORATÓRIA DE DADOS - HOTEL BOOKING DEMAND")
print("="*80)

# Importação de Bibliotecas
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
import json
warnings.filterwarnings('ignore')

# Tentar importar missingno (opcional)
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False
    print("⚠️  Biblioteca 'missingno' não encontrada. Visualizações de dados faltantes serão limitadas.")

# Configuração de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n✅ Bibliotecas importadas com sucesso!")

# ============================================================================
# CARREGAMENTO DO DATASET
# ============================================================================

print("\n" + "="*80)
print("📂 CARREGAMENTO DO DATASET")
print("="*80)

# Configuração de arquivos
zip_file_name = "hotel-booking-demand.zip"
csv_file_name = "hotel_bookings.csv"
parquet_file_name = "hotel_bookings.parquet"

# Descompactar se necessário
if os.path.exists(zip_file_name) and not os.path.exists(csv_file_name):
    print(f"\n📦 Descompactando: {zip_file_name}")
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()
    print("✅ Dataset descompactado!")
    
    # Descobrir nome do arquivo CSV
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    if csv_files:
        csv_file_name = csv_files[0]
        print(f"📄 Arquivo CSV encontrado: {csv_file_name}")

# Carregar dataset (prioriza Parquet por ser mais rápido)
if os.path.exists(parquet_file_name):
    print(f"\n📊 Carregando do Parquet...")
    df = pd.read_parquet(parquet_file_name)
    print(f"✅ Dataset carregado: {parquet_file_name}")
elif os.path.exists(csv_file_name):
    print(f"\n📊 Carregando do CSV...")
    df = pd.read_csv(csv_file_name)
    print(f"✅ Dataset CSV carregado: {csv_file_name}")
    
    # Salvar em Parquet para futuros carregamentos
    print(f"💾 Salvando em Parquet para otimização...")
    df.to_parquet(parquet_file_name, index=False)
    print(f"✅ Salvo: {parquet_file_name}")
else:
    raise FileNotFoundError(
        f"❌ Dataset não encontrado!\n"
        f"   Procurados: {parquet_file_name}, {csv_file_name}\n"
        f"   Para baixar: kaggle datasets download -d jessemostipak/hotel-booking-demand"
    )

# Informações básicas
print(f"\n📊 DIMENSÕES DO DATASET:")
print(f"   • Linhas: {df.shape[0]:,}")
print(f"   • Colunas: {df.shape[1]}")
print(f"   • Memória: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
# ============================================================================
# VISÃO GERAL E INFORMAÇÕES DO DATASET
# ============================================================================

print("\n" + "="*80)
print("� INFORMAÇÕES DO DATASET")
print("="*80)

print("\n🔍 ESTRUTURA:")
df.info()

print("\n📝 COLUNAS:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n👀 PRIMEIRAS 5 LINHAS:")
print(df.head())

print("\n📊 ÚLTIMAS 5 LINHAS:")
print(df.tail())

# ============================================================================
# ANÁLISE DE QUALIDADE DOS DADOS
# ============================================================================

print("\n" + "="*80)
print("🔍 ANÁLISE DE QUALIDADE DOS DADOS")
print("="*80)

# ----------------------------
# 1. Valores Faltantes
# ----------------------------
print("\n📊 VALORES FALTANTES:")

missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

missing_df = pd.DataFrame({
    'Valores Faltantes': missing_data,
    'Percentual (%)': missing_percent
})

missing_with_nulls = missing_df[missing_df['Valores Faltantes'] > 0]

if len(missing_with_nulls) > 0:
    print("\n   Colunas com valores faltantes:")
    display(missing_with_nulls)
    
    # Visualização matricial (se missingno disponível)
    if MISSINGNO_AVAILABLE:
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
else:
    print("\n✅ Nenhum valor faltante encontrado!")

# ----------------------------
# 2. Estatísticas Descritivas - Numéricas
# ----------------------------
print("\n" + "="*80)
print("📊 ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS NUMÉRICAS")
print("="*80)

numeric_stats = df.describe().T
numeric_stats['missing'] = missing_data
numeric_stats['missing_pct'] = missing_percent

print("\n� Resumo Estatístico:")
display(numeric_stats)

# Salvar estatísticas
numeric_stats.to_csv('numeric_statistics.csv')
print("\n💾 Estatísticas salvas em: numeric_statistics.csv")

# ----------------------------
# 3. Estatísticas Descritivas - Categóricas
# ----------------------------
print("\n" + "="*80)
print("📊 ESTATÍSTICAS DESCRITIVAS - VARIÁVEIS CATEGÓRICAS")
print("="*80)

categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\n📋 Total: {len(categorical_cols)} variáveis categóricas\n")

for col in categorical_cols:
    print(f"\n{'='*60}")
    print(f"📌 {col.upper()}")
    print(f"{'='*60}")
    
    value_counts = df[col].value_counts()
    total = len(df)
    
    print(f"   • Categorias únicas: {df[col].nunique()}")
    print(f"   • Valores faltantes: {df[col].isnull().sum()} ({df[col].isnull().sum()/total*100:.2f}%)")
    print(f"\n   Top 5 valores:")
    
    for idx, (value, count) in enumerate(value_counts.head().items(), 1):
        percentage = (count / total) * 100
        print(f"      {idx}. {value}: {count:,} ({percentage:.2f}%)")

# ----------------------------
# 4. Detecção de Outliers
# ----------------------------
print("\n" + "="*80)
print("🔍 DETECÇÃO DE OUTLIERS")
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

print("\n✅ Análise de qualidade concluída!")

# ============================================================================
# ANÁLISE DA VARIÁVEL TARGET E PADRÕES TEMPORAIS
# ============================================================================

print("\n" + "="*80)
print("🎯 ANÁLISE DA VARIÁVEL TARGET: is_canceled")
print("="*80)


cancel_counts = df['is_canceled'].value_counts()
cancel_rate = df['is_canceled'].mean() * 100

print(f"\n📊 DISTRIBUIÇÃO:")
print(f"   • Não Cancelado: {cancel_counts[0]:,} ({(cancel_counts[0]/len(df)*100):.2f}%)")
print(f"   • Cancelado: {cancel_counts[1]:,} ({(cancel_counts[1]/len(df)*100):.2f}%)")
print(f"\n🎯 Taxa Geral de Cancelamento: {cancel_rate:.2f}%")

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
    print(f"   • {hotel}: {rate:.2f}%")

# ============================================================================
# ANÁLISE TEMPORAL
# ============================================================================

print("\n" + "="*80)
print("📅 ANÁLISE TEMPORAL")
print("="*80)

# Criar coluna de data
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'] + '-' +
    df['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)

print(f"\n✅ Coluna de data criada!")
print(f"   • Período: {df['arrival_date'].min()} a {df['arrival_date'].max()}")

# ----------------------------
# Reservas ao Longo do Tempo
# ----------------------------
print("\n📊 RESERVAS AO LONGO DO TEMPO:")

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
print(f"\n   • Maior demanda: {monthly_total.idxmax()} ({monthly_total.max():,} reservas)")
print(f"   • Menor demanda: {monthly_total.idxmin()} ({monthly_total.min():,} reservas)")

# ----------------------------
# Taxa de Cancelamento Mensal
# ----------------------------
print("\n📊 TAXA DE CANCELAMENTO MENSAL:")

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

print(f"\n   • Maior taxa: {monthly_cancel.idxmax()} ({monthly_cancel.max():.2f}%)")
print(f"   • Menor taxa: {monthly_cancel.idxmin()} ({monthly_cancel.min():.2f}%)")

# ----------------------------
# Análise de Sazonalidade
# ----------------------------
print("\n📊 ANÁLISE DE SAZONALIDADE:")

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

print("\n   Por estação:")
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

print("\n✅ Análise temporal concluída!")

# ============================================================================
# ANÁLISE GEOGRÁFICA E FINANCEIRA
# ============================================================================

print("\n" + "="*80)
print("🌍 ANÁLISE GEOGRÁFICA")
print("="*80)


# ----------------------------
# Distribuição por Países
# ----------------------------
country_counts = df['country'].value_counts().head(10)

plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar', color='teal', edgecolor='black')
plt.title('Top 10 Países por Número de Reservas', fontweight='bold', fontsize=14)
plt.xlabel('País', fontsize=12)
plt.ylabel('Número de Reservas', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n📊 TOP 10 PAÍSES:")
for i, (country, count) in enumerate(country_counts.items(), 1):
    print(f"   {i:2d}. {country}: {count:,} reservas")

# Taxa de cancelamento por país
country_stats = df.groupby('country').agg({
    'is_canceled': ['count', 'mean']
}).round(3)
country_stats.columns = ['total_reservas', 'taxa_cancelamento']
country_stats = country_stats[country_stats['total_reservas'] > 100]
top_cancel_countries = country_stats.sort_values('taxa_cancelamento', ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_cancel_countries['taxa_cancelamento'].plot(kind='bar', color='orange', edgecolor='black')
plt.title('Top 10 Países com Maior Taxa de Cancelamento (>100 reservas)', 
          fontweight='bold', fontsize=14)
plt.xlabel('País', fontsize=12)
plt.ylabel('Taxa de Cancelamento', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n📊 MAIORES TAXAS DE CANCELAMENTO (países com >100 reservas):")
for i, (country, row) in enumerate(top_cancel_countries.iterrows(), 1):
    print(f"   {i:2d}. {country}: {row['taxa_cancelamento']:.3f}")

# ============================================================================
# ANÁLISE FINANCEIRA (ADR - Average Daily Rate)
# ============================================================================

print("\n" + "="*80)
print("💰 ANÁLISE DE PREÇOS (ADR - Average Daily Rate)")
print("="*80)

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribuição do ADR
axes[0, 0].hist(df['adr'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0, 0].set_title('Distribuição do ADR', fontweight='bold')
axes[0, 0].set_xlabel('ADR')
axes[0, 0].set_ylabel('Frequência')

# ADR por tipo de hotel
sns.boxplot(x='hotel', y='adr', data=df, ax=axes[0, 1])
axes[0, 1].set_title('ADR por Tipo de Hotel', fontweight='bold')
axes[0, 1].set_ylabel('ADR')

# ADR vs Cancelamento
sns.boxplot(x='is_canceled', y='adr', data=df, ax=axes[1, 0])
axes[1, 0].set_title('ADR vs Cancelamento', fontweight='bold')
axes[1, 0].set_ylabel('ADR')
axes[1, 0].set_xlabel('Cancelado (0=Não, 1=Sim)')

# Preço médio por mês
monthly_adr = df.groupby('arrival_date_month')['adr'].mean()
monthly_adr = monthly_adr.reindex(month_order)

monthly_adr.plot(kind='bar', ax=axes[1, 1], color='green', edgecolor='black')
axes[1, 1].set_title('Preço Médio (ADR) por Mês', fontweight='bold')
axes[1, 1].set_xlabel('Mês')
axes[1, 1].set_ylabel('ADR Médio')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Estatísticas do ADR
df_clean = df[df['adr'] < 1000]

print(f"\n📊 ESTATÍSTICAS DO ADR (sem outliers extremos):")
print(f"   • Média: ${df_clean['adr'].mean():.2f}")
print(f"   • Mediana: ${df_clean['adr'].median():.2f}")
print(f"   • Máximo: ${df_clean['adr'].max():.2f}")
print(f"   • Mínimo: ${df_clean['adr'].min():.2f}")

# ADR por cancelamento
adr_cancel = df[df['is_canceled'] == 1]['adr'].mean()
adr_no_cancel = df[df['is_canceled'] == 0]['adr'].mean()

print(f"\n📊 ADR POR STATUS:")
print(f"   • Cancelado: ${adr_cancel:.2f}")
print(f"   • Não Cancelado: ${adr_no_cancel:.2f}")
print(f"   • Diferença: ${abs(adr_cancel - adr_no_cancel):.2f}")

# ============================================================================
# ANÁLISE DE LEAD TIME (ANTECEDÊNCIA DA RESERVA)
# ============================================================================

print("\n" + "="*80)
print("⏰ ANÁLISE DE LEAD TIME (ANTECEDÊNCIA DA RESERVA)")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribuição do lead time
axes[0].hist(df['lead_time'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('Distribuição do Lead Time', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Lead Time (dias)', fontsize=12)
axes[0].set_ylabel('Frequência', fontsize=12)

# Lead time vs Cancelamento
sns.boxplot(x='is_canceled', y='lead_time', data=df, ax=axes[1])
axes[1].set_title('Lead Time vs Cancelamento', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Cancelado (0=Não, 1=Sim)', fontsize=12)
axes[1].set_ylabel('Lead Time (dias)', fontsize=12)

plt.tight_layout()
plt.show()

# Estatísticas detalhadas
lead_time_mean = df['lead_time'].mean()
lead_time_cancel = df[df['is_canceled'] == 1]['lead_time'].mean()
lead_time_no_cancel = df[df['is_canceled'] == 0]['lead_time'].mean()

print(f"\n📊 ESTATÍSTICAS DE LEAD TIME:")
print(f"   • Média Geral: {lead_time_mean:.1f} dias")
print(f"   • Cancelamentos: {lead_time_cancel:.1f} dias")
print(f"   • Não Cancelamentos: {lead_time_no_cancel:.1f} dias")
print(f"   • Diferença: {abs(lead_time_cancel - lead_time_no_cancel):.1f} dias")

# Correlação
correlation = df['lead_time'].corr(df['is_canceled'])
print(f"\n📈 Correlação Lead Time vs Cancelamento: {correlation:.3f}")

# ============================================================================
# ANÁLISE MULTIVARIADA
# ============================================================================

print("\n" + "="*80)
print("🔗 ANÁLISE MULTIVARIADA: LEAD TIME VS ADR VS CANCELAMENTO")
print("="*80)

df_plot = df[df['adr'] < 1000].copy()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(df_plot['lead_time'], df_plot['adr'], 
                     c=df_plot['is_canceled'], cmap='viridis', 
                     alpha=0.6, edgecolors='none')
plt.colorbar(scatter, label='Cancelado')
plt.title('Lead Time vs ADR vs Cancelamento', fontweight='bold', fontsize=14)
plt.xlabel('Lead Time (dias)', fontsize=12)
plt.ylabel('ADR', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# ANÁLISE DE SEGMENTOS E CORRELAÇÕES
# ============================================================================

plt.style.use('ggplot')
sns.set_palette("husl")

df = pd.read_parquet("hotel_bookings.parquet")
print(f"✅ Dataset carregado: {df.shape[0]:,} linhas\n")

# ----------------------------------------------------------------------------
# Análise de Segmento de Mercado e Canal de Distribuição
# ----------------------------------------------------------------------------
print("="*80)
print("ANÁLISE DE SEGMENTO DE MERCADO E CANAL DE DISTRIBUIÇÃO")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Segmento de mercado
market_segment_counts = df['market_segment'].value_counts()
axes[0, 0].bar(range(len(market_segment_counts)), market_segment_counts.values, 
               color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribuição por Segmento de Mercado', fontweight='bold')
axes[0, 0].set_xlabel('Segmento de Mercado')
axes[0, 0].set_ylabel('Número de Reservas')
axes[0, 0].set_xticks(range(len(market_segment_counts)))
axes[0, 0].set_xticklabels(market_segment_counts.index, rotation=45, ha='right')

# Taxa de cancelamento por segmento
cancel_by_segment = df.groupby('market_segment')['is_canceled'].mean() * 100
axes[0, 1].bar(range(len(cancel_by_segment)), cancel_by_segment.values, 
               color='salmon', edgecolor='black')
axes[0, 1].set_title('Taxa de Cancelamento por Segmento', fontweight='bold')
axes[0, 1].set_xlabel('Segmento de Mercado')
axes[0, 1].set_ylabel('Taxa de Cancelamento (%)')
axes[0, 1].set_xticks(range(len(cancel_by_segment)))
axes[0, 1].set_xticklabels(cancel_by_segment.index, rotation=45, ha='right')

# Canal de distribuição
distribution_channel_counts = df['distribution_channel'].value_counts()
axes[1, 0].bar(range(len(distribution_channel_counts)), distribution_channel_counts.values, 
               color='lightgreen', edgecolor='black')
axes[1, 0].set_title('Distribuição por Canal de Distribuição', fontweight='bold')
axes[1, 0].set_xlabel('Canal de Distribuição')
axes[1, 0].set_ylabel('Número de Reservas')
axes[1, 0].set_xticks(range(len(distribution_channel_counts)))
axes[1, 0].set_xticklabels(distribution_channel_counts.index, rotation=45, ha='right')

# Taxa de cancelamento por canal
cancel_by_channel = df.groupby('distribution_channel')['is_canceled'].mean() * 100
axes[1, 1].bar(range(len(cancel_by_channel)), cancel_by_channel.values, 
               color='orange', edgecolor='black')
axes[1, 1].set_title('Taxa de Cancelamento por Canal', fontweight='bold')
axes[1, 1].set_xlabel('Canal de Distribuição')
axes[1, 1].set_ylabel('Taxa de Cancelamento (%)')
axes[1, 1].set_xticks(range(len(cancel_by_channel)))
axes[1, 1].set_xticklabels(cancel_by_channel.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

print("\n📊 SEGMENTOS DE MERCADO:")
for segment, count in market_segment_counts.items():
    cancel_rate = cancel_by_segment[segment]
    print(f"   {segment}: {count:,} reservas (cancelamento: {cancel_rate:.2f}%)")

# ----------------------------------------------------------------------------
# Análise de Tipos de Cliente
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ANÁLISE DE TIPOS DE CLIENTE")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Distribuição por tipo
customer_type_counts = df['customer_type'].value_counts()
axes[0].bar(range(len(customer_type_counts)), customer_type_counts.values, 
            color='lightcoral', edgecolor='black')
axes[0].set_title('Distribuição por Tipo de Cliente', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Tipo de Cliente', fontsize=12)
axes[0].set_ylabel('Número de Reservas', fontsize=12)
axes[0].set_xticks(range(len(customer_type_counts)))
axes[0].set_xticklabels(customer_type_counts.index, rotation=45, ha='right')

# Taxa de cancelamento por tipo
cancel_by_customer = df.groupby('customer_type')['is_canceled'].mean() * 100
axes[1].bar(range(len(cancel_by_customer)), cancel_by_customer.values, 
            color='mediumpurple', edgecolor='black')
axes[1].set_title('Taxa de Cancelamento por Tipo de Cliente', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Tipo de Cliente', fontsize=12)
axes[1].set_ylabel('Taxa de Cancelamento (%)', fontsize=12)
axes[1].set_xticks(range(len(cancel_by_customer)))
axes[1].set_xticklabels(cancel_by_customer.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Análise detalhada
customer_analysis = df.groupby('customer_type').agg({
    'is_canceled': 'mean',
    'adr': 'mean',
    'lead_time': 'mean',
    'total_of_special_requests': 'mean'
}).round(3)

customer_analysis.columns = ['taxa_cancelamento', 'adr_medio', 'lead_time_medio', 'solicitacoes_especiais']
customer_analysis['taxa_cancelamento'] = customer_analysis['taxa_cancelamento'] * 100

print("\n📊 ANÁLISE DETALHADA POR TIPO DE CLIENTE:")
display(customer_analysis)

# ----------------------------------------------------------------------------
# Matriz de Correlações
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("MATRIZ DE CORRELAÇÕES")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Correlações com target
target_correlations = correlation_matrix['is_canceled'].sort_values(ascending=False)

print("\n🔝 PRINCIPAIS CORRELAÇÕES COM CANCELAMENTO:")
display(target_correlations)

# Heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            annot_kws={'size': 8}, fmt='.2f', linewidths=0.5)
plt.title('Matriz de Correlação entre Variáveis Numéricas', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

# Correlações fortes
strong_correlations = target_correlations[(abs(target_correlations) > 0.1) & (target_correlations != 1.0)]
print("\n🎯 VARIÁVEIS COM CORRELAÇÃO FORTE (|corr| > 0.1):")
display(strong_correlations)

# ----------------------------------------------------------------------------
# Análise Multivariada Avançada
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("ANÁLISE MULTIVARIADA AVANÇADA")
print("="*80)

# Hotel vs Lead Time vs Cancelamento
plt.figure(figsize=(12, 6))
sns.boxplot(x='hotel', y='lead_time', hue='is_canceled', data=df)
plt.title('Tipo de Hotel vs Lead Time vs Cancelamento', fontweight='bold', fontsize=14)
plt.ylabel('Lead Time (dias)', fontsize=12)
plt.xlabel('Tipo de Hotel', fontsize=12)
plt.legend(title='Cancelado', labels=['Não', 'Sim'])
plt.tight_layout()
plt.show()

# Cancelamentos por país e hotel
pivot_table = df.pivot_table(values='is_canceled',
                            index='country',
                            columns='hotel',
                            aggfunc='mean',
                            fill_value=0)

top_countries = df['country'].value_counts().head(15).index
pivot_table_filtered = pivot_table.loc[top_countries]

plt.figure(figsize=(14, 8))
pivot_table_filtered.plot(kind='bar', figsize=(14, 8), edgecolor='black')
plt.title('Taxa de Cancelamento por País e Tipo de Hotel (Top 15)', fontweight='bold', fontsize=14)
plt.ylabel('Taxa de Cancelamento', fontsize=12)
plt.xlabel('País', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Tipo de Hotel')
plt.tight_layout()
plt.show()

