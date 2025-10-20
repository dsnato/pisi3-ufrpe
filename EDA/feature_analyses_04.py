import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Caminho relativo partindo da pasta src
df = pd.read_csv(os.path.join("..", "data", "hotel_bookings.csv"))

# Configuração de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

print("💰 ANÁLISE DE PREÇOS E TARIFAS")

plt.figure(figsize=(15, 10))

# Distribuição do ADR (Average Daily Rate)
plt.subplot(2, 2, 1)
sns.histplot(df['adr'], bins=50, kde=True, color='purple')
plt.title('Distribuição do ADR (Average Daily Rate)')
plt.xlabel('ADR')
plt.ylabel('Frequência')

# ADR por tipo de hotel
plt.subplot(2, 2, 2)
sns.boxplot(x='hotel', y='adr', data=df)
plt.title('ADR por Tipo de Hotel')
plt.ylabel('ADR')
plt.xlabel('Tipo de Hotel')

# ADR vs Cancelamento
plt.subplot(2, 2, 3)
sns.boxplot(x='is_canceled', y='adr', data=df)
plt.title('ADR vs Cancelamento')
plt.ylabel('ADR')
plt.xlabel('Cancelado (0=Não, 1=Sim)')

# Preço médio por mês
plt.subplot(2, 2, 4)
monthly_adr = df.groupby('arrival_date_month')['adr'].mean()
monthly_adr = monthly_adr.reindex(month_order)
monthly_adr.plot(kind='bar', color='green')
plt.title('Preço Médio (ADR) por Mês')
plt.xlabel('Mês')
plt.ylabel('ADR Médio')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Remover outliers extremos para melhor visualização
df_clean = df[df['adr'] < 1000]

print(f"📊 Estatísticas do ADR (limpo):")
print(f"Média: ${df_clean['adr'].mean():.2f}")
print(f"Mediana: ${df_clean['adr'].median():.2f}")
print(f"Máximo: ${df_clean['adr'].max():.2f}")
print(f"Mínimo: ${df_clean['adr'].min():.2f}")

print("⏰ ANÁLISE DE LEAD TIME (ANTECEDÊNCIA DA RESERVA)")

plt.figure(figsize=(15, 5))

# Distribuição do lead time
plt.subplot(1, 2, 1)
sns.histplot(df['lead_time'], bins=50, kde=True, color='blue')
plt.title('Distribuição do Lead Time')
plt.xlabel('Lead Time (dias)')
plt.ylabel('Frequência')

# Lead time vs Cancelamento
plt.subplot(1, 2, 2)
sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.title('Lead Time vs Cancelamento')
plt.xlabel('Cancelado (0=Não, 1=Sim)')
plt.ylabel('Lead Time (dias)')

plt.tight_layout()
plt.show()

# Análise mais detalhada
print("📊 Estatísticas de Lead Time:")
print(f"Lead Time Médio: {df['lead_time'].mean():.1f} dias")
print(f"Lead Time Médio para cancelamentos: {df[df['is_canceled']==1]['lead_time'].mean():.1f} dias")
print(f"Lead Time Médio para não cancelamentos: {df[df['is_canceled']==0]['lead_time'].mean():.1f} dias")

# Correlação entre lead time e cancelamento
correlation = df['lead_time'].corr(df['is_canceled'])
print(f"📈 Correlação entre Lead Time e Cancelamento: {correlation:.3f}")

print("🏢 ANÁLISE DE SEGMENTO DE MERCADO E CANAL DE DISTRIBUIÇÃO")

plt.figure(figsize=(15, 10))

# Distribuição por segmento de mercado
plt.subplot(2, 2, 1)
market_segment_counts = df['market_segment'].value_counts()
market_segment_counts.plot(kind='bar', color='skyblue')
plt.title('Distribuição por Segmento de Mercado')
plt.xlabel('Segmento de Mercado')
plt.ylabel('Número de Reservas')
plt.xticks(rotation=45)

# Taxa de cancelamento por segmento
plt.subplot(2, 2, 2)
cancel_by_segment = df.groupby('market_segment')['is_canceled'].mean() * 100
cancel_by_segment.plot(kind='bar', color='salmon')
plt.title('Taxa de Cancelamento por Segmento de Mercado')
plt.xlabel('Segmento de Mercado')
plt.ylabel('Taxa de Cancelamento (%)')
plt.xticks(rotation=45)

# Distribuição por canal de distribuição
plt.subplot(2, 2, 3)
distribution_channel_counts = df['distribution_channel'].value_counts()
distribution_channel_counts.plot(kind='bar', color='lightgreen')
plt.title('Distribuição por Canal de Distribuição')
plt.xlabel('Canal de Distribuição')
plt.ylabel('Número de Reservas')
plt.xticks(rotation=45)

# Taxa de cancelamento por canal
plt.subplot(2, 2, 4)
cancel_by_channel = df.groupby('distribution_channel')['is_canceled'].mean() * 100
cancel_by_channel.plot(kind='bar', color='orange')
plt.title('Taxa de Cancelamento por Canal de Distribuição')
plt.xlabel('Canal de Distribuição')
plt.ylabel('Taxa de Cancelamento (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("👥 ANÁLISE DE TIPOS DE CLIENTE")

plt.figure(figsize=(15, 5))

# Distribuição por tipo de cliente
plt.subplot(1, 2, 1)
customer_type_counts = df['customer_type'].value_counts()
customer_type_counts.plot(kind='bar', color='lightcoral')
plt.title('Distribuição por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Número de Reservas')
plt.xticks(rotation=45)

# Taxa de cancelamento por tipo de cliente
plt.subplot(1, 2, 2)
cancel_by_customer = df.groupby('customer_type')['is_canceled'].mean() * 100
cancel_by_customer.plot(kind='bar', color='mediumpurple')
plt.title('Taxa de Cancelamento por Tipo de Cliente')
plt.xlabel('Tipo de Cliente')
plt.ylabel('Taxa de Cancelamento (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Análise detalhada por tipo de cliente
customer_analysis = df.groupby('customer_type').agg({
    'is_canceled': 'mean',
    'adr': 'mean',
    'lead_time': 'mean',
    'total_of_special_requests': 'mean'
}).round(3)

customer_analysis.columns = ['taxa_cancelamento', 'adr_medio', 'lead_time_medio', 'solicitacoes_especiais_media']
display(customer_analysis)
