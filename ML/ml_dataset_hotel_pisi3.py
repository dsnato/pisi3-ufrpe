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