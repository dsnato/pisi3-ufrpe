import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Bibliotecas importadas com sucesso!")

# Carregar o dataset
df = pd.read_csv('hotel_bookings.csv')

# Primeira visualização
print("📊 DIMENSÕES DO DATASET:")
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n👀 PRIMEIRAS LINHAS:")
display(df.head())

print("\n📋 INFORMAÇÕES GERAIS:")
df.info()

