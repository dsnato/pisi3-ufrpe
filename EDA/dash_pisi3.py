# -*- coding: utf-8 -*-
"""
PARTE 1: BASE - CONFIGURAÇÃO E CARREGAMENTO DE DADOS
Dashboard Hotel Booking Analysis
"""

print("🚀 PARTE 1: INICIANDO - CONFIGURAÇÃO BASE E CARREGAMENTO DE DADOS")

# ============================================================================
# CONFIGURAÇÃO INICIAL E IMPORTAÇÕES
# ============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✅ Dependências básicas carregadas!")

# ============================================================================
# PALETA DE CORES
# ============================================================================

COLORS = {
    'primary': '#132F3B',
    'secondary': '#0162B3',
    'accent': '#FF4F19',
    'background': '#EFEFF0',
    'dark': '#132F3B',
    'white': '#FFFFFF',
    'text': '#132F3B'
}

print("✅ Paleta de cores configurada!")

# ============================================================================
# CARREGAMENTO DOS DADOS
# ============================================================================

print("\n" + "=" * 80)
print("📂 CARREGAMENTO DOS DADOS")
print("=" * 80)


def load_and_preprocess_data():
    """Carrega e pré-processa os dados do hotel booking"""

    dataset_paths = ['ML/data/hotel_bookings.csv', 'hotel_bookings.csv', 'ML\\data\\hotel_bookings.csv']
    
    df = None
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"📊 Carregando dataset de: {dataset_path}...")
            df = pd.read_csv(dataset_path)
            print(f"✅ Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")
            return df
    
    print("⚠️  Arquivo 'hotel_bookings.csv' não encontrado")
    print("📂 Procurado em: ML/data/hotel_bookings.csv e hotel_bookings.csv")
    return None


# Carregar dados
df = load_and_preprocess_data()

if df is None:
    print("🔄 Criando dados de demonstração...")
    np.random.seed(42)
    n_samples = 10000
    
    print("⚠️  MODO DEMONSTRAÇÃO: Usando dataset sintético de 10.000 registros")
    print("💡 Para usar os 120.000 registros reais, certifique-se que 'hotel_bookings.csv' está em ML/data/")
    
    profiles = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
    
    demo_data = {
        'hotel': np.where(profiles == 0, 'City Hotel', 
                         np.where(profiles == 1, 
                                np.random.choice(['Resort Hotel', 'City Hotel'], n_samples, p=[0.7, 0.3]),
                                np.random.choice(['City Hotel', 'Resort Hotel'], n_samples))),
        'is_canceled': np.where(profiles == 0, 
                               np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
                               np.where(profiles == 1,
                                       np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                                       np.random.choice([0, 1], n_samples, p=[0.65, 0.35]))),
        'lead_time': np.where(profiles == 0, 
                             np.random.exponential(20, n_samples).astype(int),
                             np.where(profiles == 1,
                                     np.random.exponential(60, n_samples).astype(int),
                                     np.random.exponential(90, n_samples).astype(int))),
        'adr': np.abs(np.where(profiles == 0, 
                              np.random.normal(130, 25, n_samples),
                              np.where(profiles == 1,
                                      np.random.normal(100, 20, n_samples),
                                      np.random.normal(70, 15, n_samples)))),
        'adults': np.where(profiles == 1, 2, np.random.choice([1, 2], n_samples)),
        'children': np.where(profiles == 1, 
                            np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
                            0),
        'babies': np.where(profiles == 1,
                          np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                          0),
        'arrival_date_month': np.random.choice(['January', 'February', 'March', 'April', 'May', 'June',
                                               'July', 'August', 'September', 'October', 'November', 'December'],
                                              n_samples),
        'arrival_date_week_number': np.random.randint(1, 53, n_samples),
        'arrival_date_day_of_month': np.random.randint(1, 29, n_samples),
        'country': np.random.choice(['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA'], n_samples),
        'market_segment': np.where(profiles == 0, 
                                  np.random.choice(['Corporate', 'Online TA'], n_samples, p=[0.7, 0.3]),
                                  'Online TA'),
        'distribution_channel': np.random.choice(['TA/TO', 'Direct', 'Corporate'], n_samples),
        'is_repeated_guest': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'previous_cancellations': np.random.poisson(0.1, n_samples),
        'previous_bookings_not_canceled': np.random.poisson(0.2, n_samples),
        'reserved_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'assigned_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples),
        'booking_changes': np.random.poisson(0.3, n_samples),
        'customer_type': 'Transient',
        'deposit_type': 'No Deposit',
        'stays_in_weekend_nights': np.where(profiles == 1, 
                                           np.random.choice([1, 2], n_samples),
                                           np.random.choice([0, 1, 2], n_samples)),
        'stays_in_week_nights': np.where(profiles == 0,
                                        np.random.choice([1, 2, 3], n_samples),
                                        np.where(profiles == 1,
                                                np.random.choice([3, 4, 5, 7], n_samples),
                                                np.random.choice([1, 2, 3], n_samples))),
        'required_car_parking_spaces': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'total_of_special_requests': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        'agent': np.random.choice([0, 1, 9, 240], n_samples),
        'company': np.random.choice([0, 40, 223], n_samples)
    }

    df = pd.DataFrame(demo_data)
    print("✅ Dados de demonstração criados com 3 perfis distintos de clientes!")

print("\n" + "=" * 80)
print("🎉 PARTE 1 CONCLUÍDA COM SUCESSO!")
print("=" * 80)
print(f"✅ Dataset carregado: {len(df):,} registros")
print(f"✅ Colunas disponíveis: {len(df.columns)}")
print(f"✅ Paleta de cores configurada")
print("📊 Dados prontos para análise!")
