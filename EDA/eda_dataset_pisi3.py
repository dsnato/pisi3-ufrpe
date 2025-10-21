# -*- coding: utf-8 -*-
"""
EDA - Hotel Booking Demand Dataset
Commit 1: Configuração Inicial e Carregamento dos Dados
"""

# ============================================================================
# SETUP INICIAL E CARREGAMENTO DE DADOS
# ============================================================================

print("="*80)
print("CONFIGURAÇÃO INICIAL E CARREGAMENTO DOS DADOS")
print("="*80)

# Importação de Bibliotecas
import os
import zipfile
import json
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

print("✅ Bibliotecas importadas com sucesso!")

# ----------------------------------------------------------------------------
# Download e Descompactação do Dataset
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("PREPARANDO DATASET")
print("="*80)

zip_file_name = "hotel-booking-demand.zip"

if not os.path.exists(zip_file_name):
    print("⚠️  Arquivo não encontrado localmente.")
    print("📥 Para baixar o dataset, execute no terminal:")
    print("   kaggle datasets download -d jessemostipak/hotel-booking-demand")
else:
    print(f"✅ Arquivo '{zip_file_name}' encontrado!")

# Descompactar dataset
if os.path.exists(zip_file_name):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall()
    print("✅ Dataset descompactado com sucesso!")
    
    csv_files = [f for f in os.listdir() if f.endswith('.csv')]
    csv_file_name = csv_files[0] if csv_files else "hotel_bookings.csv"
    print(f"📄 Arquivo CSV: {csv_file_name}")
else:
    csv_file_name = "hotel_bookings.csv"

# ----------------------------------------------------------------------------
# Carregamento e Conversão para Parquet
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("CARREGANDO DATASET")
print("="*80)

parquet_file_name = "hotel_bookings.parquet"

if os.path.exists(parquet_file_name):
    df = pd.read_parquet(parquet_file_name)
    print(f"✅ Dataset carregado do Parquet: {parquet_file_name}")
elif os.path.exists(csv_file_name):
    print(f"📊 Carregando CSV: {csv_file_name}")
    df = pd.read_csv(csv_file_name)
    print(f"✅ Dataset CSV carregado!")
    
    df.to_parquet(parquet_file_name, index=False)
    print(f"💾 Dataset salvo em Parquet: {parquet_file_name}")
else:
    raise FileNotFoundError(f"❌ Arquivo não encontrado: {parquet_file_name} ou {csv_file_name}")

# ----------------------------------------------------------------------------
# Visão Geral do Dataset
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("VISÃO GERAL DO DATASET")
print("="*80)

print(f"\n📊 DIMENSÕES:")
print(f"   Linhas: {df.shape[0]:,}")
print(f"   Colunas: {df.shape[1]}")

print("\n👀 PRIMEIRAS 5 LINHAS:")
display(df.head())

print("\n📋 ÚLTIMAS 5 LINHAS:")
display(df.tail())

print("\n🔍 INFORMAÇÕES GERAIS:")
df.info()

print("\n📝 COLUNAS DO DATASET:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

print("\n" + "="*80)
print("✅ CONFIGURAÇÃO CONCLUÍDA!")
print("="*80)