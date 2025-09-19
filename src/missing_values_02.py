import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from IPython.display import display

df = pd.read_csv('hotel_bookings.csv')

print("🔍 ANÁLISE DE VALORES FALTANTES:")

# Análise percentual de valores nulos
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({
    'Valores Faltantes': missing_data,
    'Percentual (%)': missing_percent
})
display(missing_df[missing_df['Valores Faltantes'] > 0])

# Visualização matricial de valores faltantes
plt.figure(figsize=(12, 6))
msno.matrix(df)
plt.title('Matriz de Valores Faltantes', fontsize=16, fontweight='bold')
plt.show()

# Heatmap de correlação de missing values
plt.figure(figsize=(10, 6))
msno.heatmap(df)
plt.title('Correlação entre Valores Faltantes', fontsize=16, fontweight='bold')
plt.show()
