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

# ============================================================================
# TREINAMENTO DE MODELOS DE CLASSIFICAÇÃO
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤖 TREINAMENTO DE MODELOS DE CLASSIFICAÇÃO")
print("="*80)

# ----------------------------------------------------------------------------
# Carregar dados pré-processados
# ----------------------------------------------------------------------------
print("\n📂 Carregando dados pré-processados...")

X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')
preprocessor = joblib.load('preprocessor.pkl')

print(f"   ✅ Treino: {X_train.shape[0]:,} amostras")
print(f"   ✅ Teste: {X_test.shape[0]:,} amostras")

# ----------------------------------------------------------------------------
# Definir Modelos
# ----------------------------------------------------------------------------
print("\n📋 Definindo modelos...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, 
                                  eval_metric='logloss', n_jobs=1)
}

print(f"   Total de modelos: {len(models)}")
for name in models.keys():
    print(f"      • {name}")

# ----------------------------------------------------------------------------
# Treinar e Avaliar Modelos
# ----------------------------------------------------------------------------
print("\n🔍 Treinando modelos...")
print("   ⏱️ Estimativa: 5-10 minutos\n")

results = {}

for idx, (name, model) in enumerate(models.items(), 1):
    print(f"   [{idx}/{len(models)}] {name}...")
    
    try:
        # Criar pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Treinar
        pipeline.fit(X_train, y_train)
        
        # Prever
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Armazenar
        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"      ✅ Acurácia: {accuracy:.4f} | F1: {f1:.4f}")
        
    except Exception as e:
        print(f"      ❌ Erro: {str(e)}")
        continue

print(f"\n   ✅ {len(results)} modelos treinados com sucesso!")

# ----------------------------------------------------------------------------
# Comparação de Resultados
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print("📊 COMPARAÇÃO DE MODELOS")
print("="*80)

results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Acurácia': [results[name]['accuracy'] for name in results],
    'Precisão': [results[name]['precision'] for name in results],
    'Recall': [results[name]['recall'] for name in results],
    'F1-Score': [results[name]['f1'] for name in results]
}).sort_values('F1-Score', ascending=False)

print("\n📊 MÉTRICAS DE DESEMPENHO:")
from IPython.display import display
display(results_df)

# Identificar melhor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_f1 = results_df.iloc[0]['F1-Score']

print(f"\n🎯 MELHOR MODELO: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")

# ----------------------------------------------------------------------------
# Análise Detalhada do Melhor Modelo
# ----------------------------------------------------------------------------
print("\n" + "="*80)
print(f"🔍 ANÁLISE DETALHADA: {best_model_name}")
print("="*80)

y_pred_best = results[best_model_name]['y_pred']

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_best)
print("\n📋 MATRIZ DE CONFUSÃO:")
print(cm)
print(f"\n   Verdadeiros Negativos: {cm[0,0]:,}")
print(f"   Falsos Positivos: {cm[0,1]:,}")
print(f"   Falsos Negativos: {cm[1,0]:,}")
print(f"   Verdadeiros Positivos: {cm[1,1]:,}")

# Visualizar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Não Cancelado', 'Cancelado'],
            yticklabels=['Não Cancelado', 'Cancelado'])
plt.title(f'Matriz de Confusão - {best_model_name}', fontweight='bold', fontsize=14)
plt.ylabel('Valor Real')
plt.xlabel('Valor Previsto')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n   ✅ Gráfico salvo: confusion_matrix.png")
plt.show()

# Relatório de Classificação
print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_pred_best, 
                          target_names=['Não Cancelado', 'Cancelado']))

# ----------------------------------------------------------------------------
# Comparação Visual dos Modelos
# ----------------------------------------------------------------------------
print("\n📈 Criando comparação visual...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Acurácia
axes[0, 0].barh(results_df['Modelo'], results_df['Acurácia'], color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Acurácia')
axes[0, 0].set_title('Acurácia por Modelo', fontweight='bold')
axes[0, 0].set_xlim(0, 1)

# Precisão
axes[0, 1].barh(results_df['Modelo'], results_df['Precisão'], color='lightgreen', edgecolor='black')
axes[0, 1].set_xlabel('Precisão')
axes[0, 1].set_title('Precisão por Modelo', fontweight='bold')
axes[0, 1].set_xlim(0, 1)

# Recall
axes[1, 0].barh(results_df['Modelo'], results_df['Recall'], color='salmon', edgecolor='black')
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_title('Recall por Modelo', fontweight='bold')
axes[1, 0].set_xlim(0, 1)

# F1-Score
axes[1, 1].barh(results_df['Modelo'], results_df['F1-Score'], color='gold', edgecolor='black')
axes[1, 1].set_xlabel('F1-Score')
axes[1, 1].set_title('F1-Score por Modelo', fontweight='bold')
axes[1, 1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("   ✅ Gráfico salvo: model_comparison.png")
plt.show()

# ----------------------------------------------------------------------------
# Salvar Resultados
# ----------------------------------------------------------------------------
print("\n💾 Salvando resultados...")

# Salvar todos os modelos
for name, data in results.items():
    filename = f"model_{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(data['model'], filename)
    print(f"   ✅ {filename}")

# Salvar melhor modelo separadamente
joblib.dump(results[best_model_name]['model'], 'best_model.pkl')
print(f"   ✅ best_model.pkl")

# Salvar DataFrame de resultados
results_df.to_csv('model_results.csv', index=False)
print(f"   ✅ model_results.csv")

# Salvar previsões do melhor modelo
predictions_df = pd.DataFrame({
    'y_true': y_test,
    'y_pred': y_pred_best,
    'y_pred_proba': results[best_model_name]['y_pred_proba']
})
predictions_df.to_csv('best_model_predictions.csv', index=False)
print(f"   ✅ best_model_predictions.csv")

print("\n" + "="*80)
print("✅ TREINAMENTO DE MODELOS CONCLUÍDO!")
print("="*80)
print(f"\n📊 Resumo:")
print(f"   • Modelos treinados: {len(results)}")
print(f"   • Melhor modelo: {best_model_name}")
print(f"   • Melhor F1-Score: {best_f1:.4f}")
print(f"   • Arquivos salvos: {len(results) + 4}")
print("="*80)

