# -*- coding: utf-8 -*-
"""
Machine Learning - Hotel Booking Demand
Commit 1: Setup Inicial e Carregamento de Dados
"""

# ============================================================================
# SETUP E CONFIGURAÇÃO INICIAL
# ============================================================================

from IPython.display import display
import joblib
import os

# Configuração para evitar erros de multiprocessing
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 INICIANDO PROCESSO DE MACHINE LEARNING")
print("="*80)

# ----------------------------------------------------------------------------
# Carregamento de Dados
# ----------------------------------------------------------------------------
print("\n📂 [1/2] Carregando dados...")

parquet_file = 'hotel_bookings.parquet'
csv_file = 'hotel_bookings.csv'

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"✅ Dataset carregado do Parquet: {parquet_file}")
elif os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    print(f"✅ Dataset carregado do CSV: {csv_file}")
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Dataset salvo em Parquet: {parquet_file}")
else:
    # Tentar caminhos alternativos
    csv_file_alt = '../hotel_bookings.csv'
    parquet_file_alt = '../hotel_bookings.parquet'
    
    if os.path.exists(parquet_file_alt):
        df = pd.read_parquet(parquet_file_alt)
        print(f"✅ Dataset carregado: {parquet_file_alt}")
    elif os.path.exists(csv_file_alt):
        df = pd.read_csv(csv_file_alt)
        print(f"✅ Dataset carregado: {csv_file_alt}")
        df.to_parquet(parquet_file_alt, index=False)
        print(f"✅ Dataset salvo em Parquet: {parquet_file_alt}")
    else:
        raise FileNotFoundError("❌ Arquivo não encontrado!")

print(f"📊 Dataset: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

# Verificar uso de memória
memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
print(f"💾 Uso de memória: {memory_usage:.2f} MB")
if memory_usage > 500:
    print("⚠️ Dataset grande, processamento pode demorar...")

# ----------------------------------------------------------------------------
# Visão Geral dos Dados
# ----------------------------------------------------------------------------
print("\n📊 [2/2] Análise inicial dos dados...")

print("\n👀 PRIMEIRAS LINHAS:")
display(df.head())

print("\n📋 INFORMAÇÕES DO DATASET:")
df.info()

print("\n📊 ESTATÍSTICAS DESCRITIVAS:")
display(df.describe())

print("\n🎯 DISTRIBUIÇÃO DA VARIÁVEL TARGET:")
target_dist = df['is_canceled'].value_counts()
print(f"   Não Cancelado: {target_dist[0]:,} ({target_dist[0]/len(df)*100:.2f}%)")
print(f"   Cancelado: {target_dist[1]:,} ({target_dist[1]/len(df)*100:.2f}%)")

print("\n🔍 VALORES FALTANTES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Coluna': missing.index,
    'Faltantes': missing.values,
    'Percentual': missing_pct.values
})
missing_with_nulls = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)

if len(missing_with_nulls) > 0:
    print("\n   Colunas com valores faltantes:")
    display(missing_with_nulls)
else:
    print("   ✅ Nenhum valor faltante!")

print("\n" + "="*80)
print("✅ CARREGAMENTO E ANÁLISE INICIAL CONCLUÍDOS!")
print("="*80)

# Carregar dados
df = pd.read_parquet('hotel_bookings.parquet')
print("✅ Dataset carregado\n")

print("="*80)
print("🔧 PRÉ-PROCESSAMENTO DOS DADOS")
print("="*80)

# ----------------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------------
print("\n[1/4] Criando features derivadas...")

df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['has_special_request'] = (df['total_of_special_requests'] > 0).astype(int)
df['is_family'] = ((df['adults'] > 0) & ((df['children'] > 0) | (df['babies'] > 0))).astype(int)

print("   ✅ Features criadas:")
print("      • total_guests: adultos + crianças + bebês")
print("      • total_nights: noites fim de semana + semana")
print("      • has_special_request: possui pedidos especiais")
print("      • is_family: indicador de família")

# ----------------------------------------------------------------------------
# Tratamento de Valores Faltantes
# ----------------------------------------------------------------------------
print("\n[2/4] Tratando valores faltantes...")

missing_before = df.isnull().sum().sum()
print(f"   Valores faltantes antes: {missing_before:,}")

df['company'].fillna(0, inplace=True)
df['agent'].fillna(0, inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['children'].fillna(0, inplace=True)

missing_after = df.isnull().sum().sum()
print(f"   Valores faltantes depois: {missing_after:,}")
print(f"   ✅ Tratados: {missing_before - missing_after:,} valores")

# ----------------------------------------------------------------------------
# Remoção de Outliers
# ----------------------------------------------------------------------------
print("\n[3/4] Removendo outliers...")

original_size = len(df)
df = df[df['adr'] < 1000]
removed = original_size - len(df)

print(f"   Tamanho original: {original_size:,}")
print(f"   Outliers removidos: {removed:,} ({removed/original_size*100:.2f}%)")
print(f"   ✅ Dataset final: {len(df):,} linhas")

# ----------------------------------------------------------------------------
# Seleção de Features
# ----------------------------------------------------------------------------
print("\n[4/4] Selecionando features para modelagem...")

features = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'country', 'market_segment',
    'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type',
    'booking_changes', 'deposit_type', 'agent', 'company', 'customer_type',
    'adr', 'required_car_parking_spaces', 'total_of_special_requests',
    'total_guests', 'total_nights', 'has_special_request', 'is_family'
]

target = 'is_canceled'

print(f"   Total de features: {len(features)}")
print(f"   Target: {target}")

# Preparar dados
X = df[features]
y = df[target]

# ----------------------------------------------------------------------------
# Divisão Treino/Teste
# ----------------------------------------------------------------------------
print("\n📦 Dividindo dados em treino e teste...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   ✅ Treino: {X_train.shape[0]:,} amostras ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"   ✅ Teste: {X_test.shape[0]:,} amostras ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\n   📊 Distribuição target (treino):")
print(f"      Não Cancelado: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
print(f"      Cancelado: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
print(f"\n   📊 Distribuição target (teste):")
print(f"      Não Cancelado: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
print(f"      Cancelado: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")

# ----------------------------------------------------------------------------
# Pipeline de Pré-processamento
# ----------------------------------------------------------------------------
print("\n⚙️ Configurando pipeline de pré-processamento...")

categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

print(f"   📋 Colunas categóricas: {len(categorical_cols)}")
print(f"   📊 Colunas numéricas: {len(numeric_cols)}")

# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

print("   ✅ Pipeline configurado!")

# ----------------------------------------------------------------------------
# Salvar objetos processados
# ----------------------------------------------------------------------------
print("\n💾 Salvando objetos processados...")

import joblib

joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(features, 'features_list.pkl')

# Salvar também o DataFrame processado
df.to_parquet('hotel_bookings_processed.parquet', index=False)

print("   ✅ X_train.pkl")
print("   ✅ X_test.pkl")
print("   ✅ y_train.pkl")
print("   ✅ y_test.pkl")
print("   ✅ preprocessor.pkl")
print("   ✅ features_list.pkl")
print("   ✅ hotel_bookings_processed.parquet")

print("\n" + "="*80)
print("✅ PRÉ-PROCESSAMENTO CONCLUÍDO!")
print("="*80)
print(f"\n📊 Resumo:")
print(f"   • Total de amostras: {len(df):,}")
print(f"   • Features: {len(features)}")
print(f"   • Treino: {len(X_train):,}")
print(f"   • Teste: {len(X_test):,}")
print(f"   • Balanceamento: {y_train.mean()*100:.2f}% cancelamentos")
print("="*80)