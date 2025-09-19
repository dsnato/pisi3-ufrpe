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

# Primeira visualização
print("📊 DIMENSÕES DO DATASET:")
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n👀 PRIMEIRAS LINHAS:")
display(df.head())