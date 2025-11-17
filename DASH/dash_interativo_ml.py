# -*- coding: utf-8 -*-

print("🚀 PARTE 3: INICIANDO - BASE + EDA + MACHINE LEARNING")

# ============================================================================
# PARTE 1: CONFIGURAÇÃO INICIAL E IMPORTAÇÕES
# ============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Configurações de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✅ Todas as dependências carregadas (incluindo ML)!")

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

# ============================================================================
# CARREGAMENTO DOS DADOS
# ============================================================================

print("\n" + "=" * 80)
print("📂 PARTE 1: CARREGAMENTO DOS DADOS")
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

# ============================================================================
# PARTE 2: ANÁLISE EXPLORATÓRIA (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("🔍 PARTE 2: EXECUTANDO ANÁLISE EXPLORATÓRIA (EDA)")
print("=" * 80)


def perform_eda(df):
    """Executa análise exploratória completa"""

    print("📊 Realizando análise exploratória...")

    # Métricas básicas
    total_bookings = len(df)
    cancel_rate = df['is_canceled'].mean() * 100
    avg_adr = df['adr'].mean()

    print(f"   • Total de reservas: {total_bookings:,}")
    print(f"   • Taxa de cancelamento: {cancel_rate:.2f}%")
    print(f"   • ADR médio: ${avg_adr:.2f}")

    # Análise por hotel
    hotel_stats = df.groupby('hotel').agg({
        'is_canceled': 'mean',
        'adr': 'mean',
        'lead_time': 'mean'
    }).round(3)

    print(f"\n🏨 Estatísticas por tipo de hotel:")
    print(hotel_stats)

    # Análise temporal
    monthly_bookings = df['arrival_date_month'].value_counts()
    monthly_cancel = df.groupby('arrival_date_month')['is_canceled'].mean()

    # Análise geográfica
    country_stats = df['country'].value_counts().head(10)

    # Criar features derivadas para EDA
    df_eda = df.copy()
    df_eda['total_guests'] = df_eda['adults'] + df_eda['children'] + df_eda['babies']
    df_eda['total_nights'] = df_eda['stays_in_weekend_nights'] + df_eda['stays_in_week_nights']
    df_eda['has_special_request'] = (df_eda['total_of_special_requests'] > 0).astype(int)

    return {
        'df': df_eda,
        'total_bookings': total_bookings,
        'cancel_rate': cancel_rate,
        'avg_adr': avg_adr,
        'hotel_stats': hotel_stats,
        'monthly_bookings': monthly_bookings,
        'monthly_cancel': monthly_cancel,
        'country_stats': country_stats
    }


# Executar EDA
eda_results = perform_eda(df)
df = eda_results['df']

print("✅ Análise exploratória concluída!")

# ============================================================================
# PARTE 3: MACHINE LEARNING
# ============================================================================

print("\n" + "=" * 80)
print("🤖 PARTE 3: EXECUTANDO MODELAGEM DE MACHINE LEARNING")
print("=" * 80)


def prepare_ml_data(df):
    """Prepara dados para modelagem de ML"""

    print("🔧 Preparando dados para ML...")

    # Features selecionadas
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

    # Garantir que todas as features existem
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)

    if missing_features:
        print(f"⚠️  Features não encontradas: {missing_features}")

    # Criar features faltantes se necessário
    if 'is_family' not in df.columns:
        df['is_family'] = ((df['adults'] > 0) & ((df['children'] > 0) | (df['babies'] > 0))).astype(int)

    if 'total_guests' not in df.columns:
        df['total_guests'] = df['adults'].fillna(0) + df['children'].fillna(0) + df['babies'].fillna(0)

    if 'total_nights' not in df.columns:
        df['total_nights'] = df['stays_in_weekend_nights'].fillna(0) + df['stays_in_week_nights'].fillna(0)

    if 'has_special_request' not in df.columns:
        df['has_special_request'] = (df['total_of_special_requests'].fillna(0) > 0).astype(int)

    # Tratar valores faltantes apenas para colunas que existem
    if 'company' in df.columns:
        df['company'].fillna(0, inplace=True)
    if 'agent' in df.columns:
        df['agent'].fillna(0, inplace=True)
    if 'country' in df.columns:
        df['country'].fillna('Unknown', inplace=True)
    if 'children' in df.columns:
        df['children'].fillna(0, inplace=True)

    # Remover outliers de ADR
    df = df[df['adr'] < 1000].reset_index(drop=True)

    # Usar apenas features disponíveis
    X = df[available_features]
    y = df['is_canceled']

    print(f"✅ Dados preparados: {X.shape[0]:,} amostras, {X.shape[1]} features")
    print(f"   Features utilizadas: {len(available_features)}")

    return X, y, available_features


def train_ml_models(X, y):
    """Treina modelos de machine learning"""

    print("🎯 Treinando modelos...")

    # Separar colunas numéricas e categóricas
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    print(f"   • {len(numeric_cols)} features numéricas")
    print(f"   • {len(categorical_cols)} features categóricas")

    # Pré-processador
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Modelos
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=1),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=1),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Treinar e avaliar modelos
    results = {}
    feature_importance_df = None

    for name, model in models.items():
        print(f"   🚀 Treinando {name}...")

        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])

        # Treinar modelo
        pipeline.fit(X_train, y_train)

        # Fazer previsões
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            'model': pipeline,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_proba
        }

        print(f"     ✅ {name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        # Extrair importância das features para o melhor modelo
        if name == 'RandomForest' and hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
            feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = pipeline.named_steps['classifier'].feature_importances_

            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

    # Selecionar melhor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']

    print(f"🏆 Melhor modelo: {best_model_name} (F1: {results[best_model_name]['f1_score']:.4f})")

    return results, best_model, feature_importance_df, X_test, y_test


# Preparar e treinar modelos (usando amostra para velocidade)
print("\n📊 Para treinamento de ML, usando amostra de 15.000 registros (velocidade)...")
df_ml_sample = df.sample(n=min(15000, len(df)), random_state=42)
X, y, features = prepare_ml_data(df_ml_sample)
ml_results, best_model, feature_importance_df, X_test, y_test = train_ml_models(X, y)

print("✅ Modelagem de ML concluída!")

print("\n" + "=" * 80)
print("🎉 PARTE 3 CONCLUÍDA COM SUCESSO!")
print("=" * 80)
print(f"✅ {eda_results['total_bookings']:,} reservas analisadas")
print(f"✅ {len(ml_results)} modelos de ML treinados")
print(f"✅ Melhor modelo: {max(ml_results.keys(), key=lambda x: ml_results[x]['f1_score'])}")
print("📊 Dados prontos para Clustering!")
