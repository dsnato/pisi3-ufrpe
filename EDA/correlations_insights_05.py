from IPython.display import display
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

df = pd.read_csv('hotel_bookings.csv')

print("📊 MATRIZ DE CORRELAÇÕES")

# Selecionar apenas colunas numéricas para correlação
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Focar nas correlações com a variável target
target_correlations = correlation_matrix['is_canceled'].sort_values(ascending=False)

print("🔝 Principais correlações com Cancelamento:")
display(target_correlations)

# Heatmap das correlações
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            annot_kws={'size': 8}, fmt='.2f')
plt.title('Matriz de Correlação entre Variáveis Numéricas', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.show()

# Correlações mais fortes com cancelamento
strong_correlations = target_correlations[(abs(target_correlations) > 0.1) & (target_correlations != 1.0)]
print("🎯 Variáveis com maior correlação com cancelamento (|corr| > 0.1):")
display(strong_correlations)

print("🎨 ANÁLISE MULTIVARIADA AVANÇADA")

# 1. Lead Time vs ADR vs Cancelamento
plt.figure(figsize=(12, 8))
sns.scatterplot(x='lead_time', y='adr', hue='is_canceled',
                data=df[df['adr'] < 1000], alpha=0.6, palette='viridis')
plt.title('Lead Time vs ADR vs Cancelamento', fontweight='bold')
plt.xlabel('Lead Time (dias)')
plt.ylabel('ADR')
plt.legend(title='Cancelado')
plt.show()

# 2. Tipo de Hotel vs Lead Time vs Cancelamento
plt.figure(figsize=(12, 6))
sns.boxplot(x='hotel', y='lead_time', hue='is_canceled', data=df)
plt.title('Tipo de Hotel vs Lead Time vs Cancelamento', fontweight='bold')
plt.ylabel('Lead Time (dias)')
plt.xlabel('Tipo de Hotel')
plt.legend(title='Cancelado')
plt.show()

# 3. Análise de cancelamentos por país e tipo de hotel
pivot_table = df.pivot_table(values='is_canceled',
                            index='country',
                            columns='hotel',
                            aggfunc='mean',
                            fill_value=0)

# Top 15 países com maior taxa de cancelamento
top_countries = df['country'].value_counts().head(15).index
pivot_table_filtered = pivot_table.loc[top_countries]

plt.figure(figsize=(14, 8))
pivot_table_filtered.plot(kind='bar', figsize=(14, 8))
plt.title('Taxa de Cancelamento por País e Tipo de Hotel (Top 15 países)', fontweight='bold')
plt.ylabel('Taxa de Cancelamento')
plt.xlabel('País')
plt.xticks(rotation=45)
plt.legend(title='Tipo de Hotel')
plt.tight_layout()
plt.show()

print("🎯 INSIGHTS E RECOMENDAÇÕES FINAIS")

print("=" * 60)
print("PRINCIPAIS DESCOBERTAS:")
print("=" * 60)

# Insights calculados programaticamente
cancel_rate = df['is_canceled'].mean() * 100
lead_time_cancel = df[df['is_canceled'] == 1]['lead_time'].mean()
lead_time_no_cancel = df[df['is_canceled'] == 0]['lead_time'].mean()
adr_cancel = df[df['is_canceled'] == 1]['adr'].mean()
adr_no_cancel = df[df['is_canceled'] == 0]['adr'].mean()

print(f"1. 📉 Taxa geral de cancelamento: {cancel_rate:.1f}%")
print(f"2. ⏰ Reservas canceladas têm lead time MUITO maior: {lead_time_cancel:.1f} dias vs {lead_time_no_cancel:.1f} dias")
print(f"3. 💰 Reservas canceladas têm ADR ligeiramente menor: ${adr_cancel:.2f} vs ${adr_no_cancel:.2f}")
print(f"4. 🏨 City Hotel tem maior taxa de cancelamento: {cancel_by_hotel['City Hotel']:.1f}% vs {cancel_by_hotel['Resort Hotel']:.1f}%")
print(f"5. 📊 Lead Time é a variável mais correlacionada com cancelamento: {correlation_matrix['is_canceled']['lead_time']:.3f}")

print("\n" + "=" * 60)
print("RECOMENDAÇÕES PARA REDUZIR CANCELAMENTOS:")
print("=" * 60)
print("🎯 1. Implementar política diferenciada para reservas com lead time > 100 dias")
print("🎯 2. Criar campanhas especiais para grupos com alta taxa de cancelamento")
print("🎯 3. Revisar estratégia de preços para City Hotel")
print("🎯 4. Melhorar comunicação com hóspedes que fazem reservas com muita antecedência")
print("🎯 5. Oferecer benefícios para reservas não canceláveis em períodos de alta demanda")

print("\n" + "=" * 60)
print("PRÓXIMOS PASSOS PARA MODELAGEM DE ML:")
print("=" * 60)
print("🤖 1. Pré-processamento: Tratar valores faltantes em 'company', 'agent', 'country'")
print("🤖 2. Feature Engineering: Criar variáveis como 'season', 'total_guests', 'total_nights'")
print("🤖 3. Encoding: Converter variáveis categóricas usando One-Hot Encoding")
print("🤖 4. Modelagem: Testar Random Forest, XGBoost e Logistic Regression")
print("🤖 5. Otimização: Usar GridSearch para tuning de hiperparâmetros")

print("💾 SALVANDO RESULTADOS DA ANÁLISE")

# Salvar dataset com algumas transformações úteis
df.to_csv('hotel_bookings_analyzed.csv', index=False)

# Salvar estatísticas importantes
summary_stats = {
    'total_reservas': df.shape[0],
    'taxa_cancelamento': df['is_canceled'].mean(),
    'lead_time_medio': df['lead_time'].mean(),
    'adr_medio': df['adr'].mean(),
    'hotel_counts': df['hotel'].value_counts().to_dict(),
    'top_countries': df['country'].value_counts().head(5).to_dict()
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=4)

print("✅ Análise concluída e resultados salvos!")
print("📁 Arquivos gerados:")
print("   - hotel_bookings_analyzed.csv")
print("   - analysis_summary.json")
print("   - Gráficos e visualizações")
