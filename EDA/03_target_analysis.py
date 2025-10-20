# Importação de todas as bibliotecas
import os
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Bibliotecas importadas com sucesso!")

# Caminho relativo partindo da pasta src
df = pd.read_csv(os.path.join("..", "data", "hotel_bookings.csv"))

print("🎯 ANÁLISE DA VARIÁVEL TARGET - is_canceled")

# Distribuição de cancelamentos
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
cancel_counts = df['is_canceled'].value_counts()
cancel_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c']) # Alterado para gráfico de barras
plt.title('Distribuição de Cancelamentos', fontweight='bold')
plt.xlabel('Status de Cancelamento')
plt.ylabel('Número de Reservas')
plt.xticks(ticks=[0, 1], labels=['Não Cancelado', 'Cancelado'], rotation=0)


plt.subplot(1, 2, 2)
cancel_by_hotel = df.groupby('hotel')['is_canceled'].mean() * 100
cancel_by_hotel.plot(kind='bar', color=['#3498db', '#9b59b6'])
plt.title('Taxa de Cancelamento por Tipo de Hotel')
plt.ylabel('Taxa de Cancelamento (%)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print(f"📊 Taxa Geral de Cancelamento: {df['is_canceled'].mean()*100:.2f}%")
print(f"🏨 Taxa Cancelamento City Hotel: {cancel_by_hotel['City Hotel']:.2f}%")
print(f"🏖️ Taxa Cancelamento Resort Hotel: {cancel_by_hotel['Resort Hotel']:.2f}%")

print("📅 ANÁLISE TEMPORAL DAS RESERVAS")

# Criar coluna de data completa
df['arrival_date'] = pd.to_datetime(
    df['arrival_date_year'].astype(str) + '-' +
    df['arrival_date_month'] + '-' +
    df['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)

# Análise por mês
monthly_bookings = df.groupby(['arrival_date_year', 'arrival_date_month']).size().unstack(0)
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_bookings = monthly_bookings.reindex(month_order)

plt.figure(figsize=(15, 8))
monthly_bookings.plot(kind='bar', figsize=(15, 6))
plt.title('Reservas por Mês e Ano', fontweight='bold', fontsize=14)
plt.xlabel('Mês')
plt.ylabel('Número de Reservas')
plt.xticks(rotation=45)
plt.legend(title='Ano')
plt.tight_layout()
plt.show()

# Taxa de cancelamento mensal
monthly_cancel = df.groupby('arrival_date_month')['is_canceled'].mean() * 100
monthly_cancel = monthly_cancel.reindex(month_order)

plt.figure(figsize=(12, 6))
monthly_cancel.plot(kind='line', marker='o', color='red')
plt.title('Taxa de Cancelamento por Mês', fontweight='bold', fontsize=14)
plt.xlabel('Mês')
plt.ylabel('Taxa de Cancelamento (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

print("🌎 ANÁLISE GEOGRÁFICA - DISTRIBUIÇÃO POR PAÍSES")

# Top 10 países com mais reservas
country_counts = df['country'].value_counts().head(10)

plt.figure(figsize=(12, 6))
country_counts.plot(kind='bar', color='teal')
plt.title('Top 10 Países por Número de Reservas', fontweight='bold', fontsize=14)
plt.xlabel('País')
plt.ylabel('Número de Reservas')
plt.xticks(rotation=45)
plt.show()

# Taxa de cancelamento por país (apenas países com +100 reservas)
country_stats = df.groupby('country').agg({
    'is_canceled': ['count', 'mean']
}).round(3)
country_stats.columns = ['total_reservas', 'taxa_cancelamento']
country_stats = country_stats[country_stats['total_reservas'] > 100]
top_cancel_countries = country_stats.sort_values('taxa_cancelamento', ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_cancel_countries['taxa_cancelamento'].plot(kind='bar', color='orange')
plt.title('Top 10 Países com Maior Taxa de Cancelamento (>100 reservas)', fontweight='bold', fontsize=14)
plt.xlabel('País')
plt.ylabel('Taxa de Cancelamento')
plt.xticks(rotation=45)
plt.show()
