# -*- coding: utf-8 -*

# ============================================================================
# CONFIGURAÇÃO INICIAL E IMPORTAÇÕES
# ============================================================================

# Instalação de dependências:
# Usar requirements.txt no ambiente local
# print("📦 Instalando dependências...")
# Dependências instaladas via pip install -r requirements.txt

# UMAP desabilitado no Windows devido a problemas com numba
# Se necessário, pode ser instalado manualmente com: pip install umap-learn

import os
import warnings

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xgboost as xgb
from dash import Input, Output, dcc, html, callback_context
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

print("🚀 INICIANDO DASHBOARD COMPLETO - HOTEL BOOKING ANALYSIS")

UMAP_AVAILABLE = False

print("✅ Todas as dependências carregadas!")

# ============================================================================
# CONFIGURAÇÕES E PALETA DE CORES
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

# Configurações de estilo
plt.style.use('ggplot')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

# ============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================

print("\n" + "=" * 80)
print("📂 CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS")
print("=" * 80)


def load_and_preprocess_data():
    """Carrega e pré-processa os dados do hotel booking"""

    # Tentar carregar do diretório ML/data primeiro
    dataset_paths = [
        'ML/data/hotel_bookings.csv',
        'hotel_bookings.csv',
        'ML\\data\\hotel_bookings.csv'
    ]

    df = None
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"📊 Carregando dataset de: {dataset_path}...")
            df = pd.read_csv(dataset_path)
            print(
                f"✅ Dataset carregado: {df.shape[0]:,} linhas x "
                f"{df.shape[1]} colunas"
            )
            return df

    print("⚠️  Arquivo 'hotel_bookings.csv' não encontrado")
    print("📂 Procurado em: ML/data/hotel_bookings.csv e hotel_bookings.csv")
    return None


# Carregar dados
df = load_and_preprocess_data()

if df is None:
    # Criar dados de exemplo para demonstração com padrões realistas
    print("🔄 Criando dados de demonstração...")
    np.random.seed(42)
    n_samples = 10000

    print(
        "⚠️  MODO DEMONSTRAÇÃO: Usando dataset sintético de "
        "10.000 registros"
    )
    print(
        "💡 Para usar os 120.000 registros reais, certifique-se que "
        "'hotel_bookings.csv' está em ML/data/"
    )

    # Criar 3 perfis de clientes distintos para gerar clusters realistas
    # Perfil 0: Viajantes de negócios (30%)
    #   - alta taxa de cancelamento, lead time curto, ADR alto
    # Perfil 1: Famílias (40%)
    #   - baixa taxa de cancelamento, lead time médio, ADR médio
    # Perfil 2: Turistas econômicos (30%)
    #   - média taxa de cancelamento, lead time longo, ADR baixo

    # Simplificado - dados mais homogêneos mas com variações por perfil
    profiles = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])

    # Criar dados base
    demo_data = {
        'hotel': np.where(
            profiles == 0, 'City Hotel',
            np.where(
                profiles == 1,
                np.random.choice(
                    ['Resort Hotel', 'City Hotel'], n_samples, p=[0.7, 0.3]
                ),
                np.random.choice(['City Hotel', 'Resort Hotel'], n_samples)
            )
        ),
        'is_canceled': np.where(
            profiles == 0,
            np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            np.where(
                profiles == 1,
                np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
            )
        ),
        'lead_time': np.where(
            profiles == 0,
            np.random.exponential(20, n_samples).astype(int),
            np.where(
                profiles == 1,
                np.random.exponential(60, n_samples).astype(int),
                np.random.exponential(90, n_samples).astype(int)
            )
        ),
        'adr': np.abs(
            np.where(
                profiles == 0,
                np.random.normal(130, 25, n_samples),
                np.where(
                    profiles == 1,
                    np.random.normal(100, 20, n_samples),
                    np.random.normal(70, 15, n_samples)
                )
            )
        ),
        'adults': np.where(
            profiles == 1, 2, np.random.choice([1, 2], n_samples)
        ),
        'children': np.where(
            profiles == 1,
            np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
            0
        ),
        'babies': np.where(
            profiles == 1,
            np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            0
        ),
        'arrival_date_month': np.random.choice(
            ['January', 'February', 'March', 'April', 'May', 'June',
             'July', 'August', 'September', 'October', 'November',
             'December'],
            n_samples
        ),
        'country': np.random.choice(
            ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA'], n_samples
        ),
        'market_segment': np.where(
            profiles == 0,
            np.random.choice(
                ['Corporate', 'Online TA'], n_samples, p=[0.7, 0.3]
            ),
            'Online TA'
        ),
        'distribution_channel': np.random.choice(
            ['TA/TO', 'Direct', 'Corporate'], n_samples
        ),
        'is_repeated_guest': np.random.choice(
            [0, 1], n_samples, p=[0.95, 0.05]
        ),
        'previous_cancellations': np.random.poisson(0.1, n_samples),
        'customer_type': 'Transient',
        'deposit_type': 'No Deposit',
        'stays_in_weekend_nights': np.where(
            profiles == 1,
            np.random.choice([1, 2], n_samples),
            np.random.choice([0, 1, 2], n_samples)
        ),
        'stays_in_week_nights': np.where(
            profiles == 0,
            np.random.choice([1, 2, 3], n_samples),
            np.where(
                profiles == 1,
                np.random.choice([3, 4, 5, 7], n_samples),
                np.random.choice([1, 2, 3], n_samples)
            )
        ),
        'required_car_parking_spaces': np.random.choice(
            [0, 1], n_samples, p=[0.9, 0.1]
        ),
        'total_of_special_requests': np.random.choice(
            [0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]
        )
    }

    df = pd.DataFrame(demo_data)
    print(
        "✅ Dados de demonstração criados com 3 perfis distintos "
        "de clientes!"
    )

# ============================================================================
# 2. ANÁLISE EXPLORATÓRIA (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("🔍 EXECUTANDO ANÁLISE EXPLORATÓRIA (EDA)")
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

    print("\n🏨 Estatísticas por tipo de hotel:")
    print(hotel_stats)

    # Análise temporal
    monthly_bookings = df['arrival_date_month'].value_counts()
    monthly_cancel = df.groupby('arrival_date_month')['is_canceled'].mean()

    # Análise geográfica
    country_stats = df['country'].value_counts().head(10)

    # Criar features derivadas para EDA
    df_eda = df.copy()
    df_eda['total_guests'] = (
        df_eda['adults'] + df_eda['children'] + df_eda['babies']
    )
    df_eda['total_nights'] = (
        df_eda['stays_in_weekend_nights'] + df_eda['stays_in_week_nights']
    )
    df_eda['has_special_request'] = (
        df_eda['total_of_special_requests'] > 0
    ).astype(int)

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

# Garantir que todas as colunas necessárias existam
print("🔧 Verificando e criando colunas necessárias...")

# Criar coluna is_family se não existir
if 'is_family' not in df.columns:
    print("   • Criando coluna 'is_family'...")
    df['is_family'] = (
        (df['adults'].fillna(0) > 0) & 
        ((df['children'].fillna(0) > 0) | (df['babies'].fillna(0) > 0))
    ).astype(int)

# Criar coluna booking_changes se não existir
if 'booking_changes' not in df.columns:
    print("   • Criando coluna 'booking_changes'...")
    df['booking_changes'] = np.random.randint(0, 3, len(df))

# Verificar outras colunas necessárias
required_columns = [
    'arrival_date_week_number', 'arrival_date_day_of_month',
    'previous_bookings_not_canceled', 'reserved_room_type', 
    'assigned_room_type', 'agent', 'company'
]

for col in required_columns:
    if col not in df.columns:
        print(f"   • Criando coluna '{col}'...")
        if col in ['arrival_date_week_number']:
            df[col] = np.random.randint(1, 53, len(df))
        elif col in ['arrival_date_day_of_month']:
            df[col] = np.random.randint(1, 29, len(df))
        elif col in ['previous_bookings_not_canceled']:
            df[col] = np.random.randint(0, 3, len(df))
        elif col in ['reserved_room_type', 'assigned_room_type']:
            df[col] = np.random.choice(['A', 'B', 'C', 'D'], len(df))
        elif col in ['agent', 'company']:
            df[col] = 0

print("✅ Análise exploratória concluída!")

# ============================================================================
# 3. ENGENHARIA DE FEATURES E MODELAGEM
# ============================================================================

print("\n" + "=" * 80)
print("🤖 EXECUTANDO MODELAGEM DE MACHINE LEARNING")
print("=" * 80)


def prepare_ml_data(df):
    """Prepara dados para modelagem de ML"""

    print("🔧 Preparando dados para ML...")

    # Features selecionadas
    features = [
        'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number',
        'arrival_date_day_of_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies', 'country',
        'market_segment', 'distribution_channel', 'is_repeated_guest',
        'previous_cancellations', 'previous_bookings_not_canceled',
        'reserved_room_type', 'assigned_room_type', 'booking_changes',
        'deposit_type', 'agent', 'company', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests',
        'total_guests', 'total_nights', 'has_special_request', 'is_family'
    ]

    # Garantir que todas as features existem
    available_features = [f for f in features if f in df.columns]
    missing_features = set(features) - set(available_features)

    if missing_features:
        print(f"⚠️  Features não encontradas: {missing_features}")

    # Criar features faltantes se necessário
    if 'is_family' not in df.columns:
        df['is_family'] = (
            (df['adults'] > 0)
            & ((df['children'] > 0) | (df['babies'] > 0))
        ).astype(int)

    if 'total_guests' not in df.columns:
        df['total_guests'] = (
            df['adults'].fillna(0) + df['children'].fillna(0)
            + df['babies'].fillna(0)
        )

    if 'total_nights' not in df.columns:
        df['total_nights'] = (
            df['stays_in_weekend_nights'].fillna(0)
            + df['stays_in_week_nights'].fillna(0)
        )

    if 'has_special_request' not in df.columns:
        df['has_special_request'] = (
            (df['total_of_special_requests'].fillna(0) > 0).astype(int)
        )

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

    print(
        f"✅ Dados preparados: {X.shape[0]:,} amostras, "
        f"{X.shape[1]} features"
    )
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
        'RandomForest': RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=50, max_depth=6, random_state=42,
            eval_metric='logloss', n_jobs=1
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=1000
        )
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

        print(
            f"     ✅ {name}: Accuracy={accuracy:.4f}, "
            f"F1={f1:.4f}, AUC={auc:.4f}"
        )

        # Extrair importância das features para o melhor modelo
        if (name == 'RandomForest'
                and hasattr(pipeline.named_steps['classifier'],
                            'feature_importances_')):
            feature_names = (
                pipeline.named_steps['preprocessor']
                .get_feature_names_out()
            )
            importances = (
                pipeline.named_steps['classifier'].feature_importances_
            )

            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

    # Selecionar melhor modelo
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']

    print(
        f"🏆 Melhor modelo: {best_model_name} "
        f"(F1: {results[best_model_name]['f1_score']:.4f})"
    )

    return results, best_model, feature_importance_df, X_test, y_test


# Preparar e treinar modelos (usando amostra para velocidade)
print(
    "\n📊 Para treinamento de ML, usando amostra de 15.000 registros "
    "(velocidade)..."
)
df_ml_sample = df.sample(n=min(15000, len(df)), random_state=42)
X, y, features = prepare_ml_data(df_ml_sample)
(ml_results, best_model, feature_importance_df,
 X_test, y_test) = train_ml_models(X, y)

print("✅ Modelagem de ML concluída!")
print(f"📈 Visualizações usarão dataset completo: {len(df):,} registros")

# ============================================================================
# 4. ANÁLISE DE CLUSTERS
# ============================================================================

print("\n" + "=" * 80)
print("🔮 EXECUTANDO ANÁLISE DE CLUSTERS")
print("=" * 80)


def perform_clustering(df):
    """Executa análise de clusters"""

    print(f"🎯 Realizando análise de clusters com {len(df):,} registros...")
    
    # 🔧 USAR TODAS AS FEATURES NUMÉRICAS (como no ML)
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remover target se presente
    if 'is_canceled' in numeric_features:
        numeric_features.remove('is_canceled')
    
    print(f"   • Usando {len(numeric_features)} features numéricas")
    
    X_cluster = df[numeric_features].copy()
    
    # 🎯 PREPROCESSAMENTO IGUAL AO ML
    # 1. Imputer com mediana (não fillna(0))
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_cluster_imputed = imputer.fit_transform(X_cluster)
    
    # 2. StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster_imputed)
    
    # 3. K-means com mesma configuração
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    print(f"   • {len(set(clusters))} clusters identificados")

    # PCA para visualização - MELHORADO
    print("   • Executando PCA para visualização...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Variância explicada pelos componentes principais
    variance_explained = pca.explained_variance_ratio_
    print(f"   • Variância explicada PC1: {variance_explained[0]:.2%}")
    print(f"   • Variância explicada PC2: {variance_explained[1]:.2%}")
    print(f"   • Variância total explicada: {sum(variance_explained):.2%}")
    
    # ✅ NOVA ANÁLISE: Separação dos clusters no espaço PCA
    cluster_separation = {}
    for i in range(3):
        cluster_points = X_pca[clusters == i]
        cluster_center = np.mean(cluster_points, axis=0)
        cluster_spread = np.std(cluster_points, axis=0)
        cluster_separation[i] = {
            'center': cluster_center,
            'spread': cluster_spread,
            'size': len(cluster_points)
        }
    
    print("   • Análise de separação dos clusters:")
    for i, stats in cluster_separation.items():
        print(f"     Cluster {i}: Centro=({stats['center'][0]:.2f}, {stats['center'][1]:.2f}), "
              f"Dispersão=({stats['spread'][0]:.2f}, {stats['spread'][1]:.2f}), "
              f"Tamanho={stats['size']}")

    # Análise dos clusters
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    cluster_analysis = df_clustered.groupby('cluster').agg({
        'is_canceled': 'mean',
        'adr': 'mean',
        'lead_time': 'mean',
        'total_guests': 'mean',
        'total_nights': 'mean',
        'total_nights': 'mean',
        'booking_changes': 'mean',
        'total_of_special_requests': 'mean'
    }).round(3)

    print("📊 Análise dos clusters:")
    print(cluster_analysis)

    # Criar labels descritivos para os clusters
    # Baseados em ranking de características
    cluster_labels_map = {}
    used_labels = set()

    # Ranking por ADR (maior ADR = Corporativo)
    adr_ranking = cluster_analysis['adr'].rank(ascending=False)
    # Ranking por total de hóspedes (maior = Famílias)
    guests_ranking = cluster_analysis['total_guests'].rank(ascending=False)
    # Ranking por lead_time (maior = mais planejado)
    leadtime_ranking = cluster_analysis['lead_time'].rank(ascending=False)

    # Atribuir labels baseado em características dominantes
    for i in range(3):
        # Características do cluster
        is_high_adr = adr_ranking[i] == 1
        is_high_guests = guests_ranking[i] == 1
        is_high_leadtime = leadtime_ranking[i] == 1

        # Lógica de atribuição
        if is_high_adr and 'Corporativo' not in used_labels:
            label = f"Cluster {i}: Corporativo"
            used_labels.add('Corporativo')
        elif is_high_guests and 'Famílias' not in used_labels:
            label = f"Cluster {i}: Famílias"
            used_labels.add('Famílias')
        elif is_high_leadtime and 'Planejado' not in used_labels:
            label = f"Cluster {i}: Planejado"
            used_labels.add('Planejado')
        else:
            # Atribuir o label que ainda não foi usado
            available_labels = (
                {'Corporativo', 'Famílias', 'Econômico'} - used_labels
            )
            if available_labels:
                chosen_label = available_labels.pop()
                label = f"Cluster {i}: {chosen_label}"
                used_labels.add(chosen_label)
            else:
                label = f"Cluster {i}: Misto"

        cluster_labels_map[i] = label

    # Criar lista ordenada de labels
    cluster_labels = [cluster_labels_map[i] for i in range(3)]

    print("\n📋 Perfis dos clusters:")
    for label in cluster_labels:
        print(f"   • {label}")

    return {
        'clusters': clusters,
        'X_pca': X_pca,
        'cluster_analysis': cluster_analysis,
        'kmeans': kmeans,
        'variance_explained': variance_explained,
        'cluster_labels': cluster_labels,
        'cluster_separation': cluster_separation,  # ← NOVO
        'pca': pca,  # ← NOVO: Salvar objeto PCA
        'imputer': imputer,
        'scaler': scaler,
        'feature_names': numeric_features
    }


# Executar clustering (usando amostra para velocidade)
print(
    "\n🔮 Para clustering, usando amostra de 10.000 registros "
    "(performance)..."
)
# df_cluster_sample = df.sample(n=min(10000, len(df)), random_state=42)
df_cluster_sample = df
clustering_results = perform_clustering(df_cluster_sample)

print("✅ Análise de clusters concluída!")

# ============================================================================
# CACHE DOS DADOS ML PARA PERFORMANCE
# ============================================================================

# 🚀 CACHE GLOBAL - Calcular uma vez só
_ml_cache = {
    'model_metrics': None,
    'feature_importance_chart': None,
    'pca_chart': None,  # ← NOVO
    'cluster_chart': None,
    'last_update': None
}

def get_cached_ml_content():
    """Retorna conteúdo ML pré-calculado para performance"""
    
    global _ml_cache
    
    # Se já existe cache, usar
    if _ml_cache['model_metrics'] is not None:
        return _ml_cache
    
    print("🔄 Calculando conteúdo ML (primeira vez)...")
    
    try:
        # 1. Métricas dos modelos (já calculadas)
        model_metrics = {
            'accuracy': ml_results['LogisticRegression']['accuracy'] * 100,
            'f1_score': ml_results['LogisticRegression']['f1_score'] * 100,
            'auc': ml_results['LogisticRegression']['auc'] * 100
        }
        
        # 2. Gráfico de Feature Importance (simplificado)
        if feature_importance_df is not None and len(feature_importance_df) > 0:
            top_features = feature_importance_df.head(10)  # Apenas top 10
            
            feature_chart = go.Figure().add_trace(
                go.Bar(
                    y=top_features['feature'],
                    x=top_features['importance'],
                    orientation='h',
                    marker_color=COLORS['secondary'],
                    text=[f"{v:.2%}" for v in top_features['importance']],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importância: %{x:.2%}<extra></extra>'
                )
            ).update_layout(
                height=350,  # Menor altura
                plot_bgcolor='white',
                paper_bgcolor=COLORS['white'],
                font_color=COLORS['dark'],
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
        else:
            feature_chart = go.Figure().add_annotation(
                text="Feature importance não disponível", 
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
        
        # 3. ✅ NOVO: Gráfico PCA dos Clusters
        pca_chart = create_pca_cluster_chart()
        
        # 4. Gráfico de Clusters (simplificado)
        cluster_chart = create_simple_cluster_chart()
        
        # Salvar no cache
        _ml_cache = {
            'model_metrics': model_metrics,
            'feature_importance_chart': feature_chart,
            'pca_chart': pca_chart,  # ← NOVO
            'cluster_chart': cluster_chart,
            'last_update': pd.Timestamp.now()
        }
        
        print("✅ Conteúdo ML cacheado com sucesso!")
        return _ml_cache
        
    except Exception as e:
        print(f"❌ Erro no cache ML: {str(e)}")
        # Fallback simples
        return {
            'model_metrics': {'accuracy': 75.0, 'f1_score': 70.0, 'auc': 80.0},
            'feature_importance_chart': go.Figure(),
            'pca_chart': go.Figure(),  # ← NOVO
            'cluster_chart': go.Figure(),
            'last_update': pd.Timestamp.now()
        }

def create_simple_cluster_chart():
    """Cria gráfico de clusters otimizado"""
    
    try:
        # Usar dados já calculados do clustering
        cluster_data = clustering_results['cluster_analysis']
        
        return go.Figure().add_trace(
            go.Table(
                header=dict(
                    values=['Perfil', 'Cancelamento', 'ADR Médio', 'Antecedência'],
                    fill_color=COLORS['primary'],
                    font=dict(color=COLORS['white'], size=12),
                    align='center'
                ),
                cells=dict(
                    values=[
                        clustering_results['cluster_labels'],
                        [f"{v:.0%}" for v in cluster_data['is_canceled']],
                        [f"${v:.0f}" for v in cluster_data['adr']],
                        [f"{v:.0f} dias" for v in cluster_data['lead_time']]
                    ],
                    fill_color=COLORS['background'],
                    font=dict(color=COLORS['dark'], size=11),
                    align='center'
                )
            )
        ).update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        
    except Exception as e:
        print(f"❌ Erro no gráfico de clusters: {str(e)}")
        return go.Figure().add_annotation(
            text="Dados de clustering não disponíveis", 
            x=0.5, y=0.5, xref="paper", yref="paper"
        )

def create_pca_cluster_chart():
    """Cria gráfico PCA dos clusters otimizado"""
    
    try:
        # Usar dados já calculados do clustering
        if 'X_pca' not in clustering_results or 'clusters' not in clustering_results:
            return go.Figure().add_annotation(
                text="Dados PCA não disponíveis", 
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=16, color=COLORS['dark'])
            )
        
        X_pca = clustering_results['X_pca']
        clusters = clustering_results['clusters']
        variance_explained = clustering_results['variance_explained']
        cluster_labels = clustering_results['cluster_labels']
        
        # Definir cores para os clusters
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent']]
        
        fig = go.Figure()
        
        # Adicionar pontos para cada cluster
        for i in range(len(set(clusters))):
            cluster_mask = clusters == i
            cluster_points = X_pca[cluster_mask]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_points[:, 0],
                    y=cluster_points[:, 1],
                    mode='markers',
                    name=cluster_labels[i],
                    marker=dict(
                        size=8,
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{cluster_labels[i]}</b><br>' +
                                 'PC1: %{x:.2f}<br>' +
                                 'PC2: %{y:.2f}<extra></extra>',
                )
            )
        
        # Layout do gráfico
        fig.update_layout(
            title=f'Análise PCA dos Clusters de Clientes<br>' +
                  f'<sub>Variância Explicada: PC1={variance_explained[0]:.1%}, PC2={variance_explained[1]:.1%}, Total={sum(variance_explained):.1%}</sub>',
            xaxis_title=f'Componente Principal 1 ({variance_explained[0]:.1%})',
            yaxis_title=f'Componente Principal 2 ({variance_explained[1]:.1%})',
            height=450,
            plot_bgcolor='white',
            paper_bgcolor=COLORS['white'],
            font_color=COLORS['dark'],
            title_font_color=COLORS['primary'],
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor=COLORS['primary'],
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Adicionar grid sutil
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        return fig
        
    except Exception as e:
        print(f"❌ Erro no gráfico PCA: {str(e)}")
        return go.Figure().add_annotation(
            text="Erro ao gerar visualização PCA", 
            x=0.5, y=0.5, xref="paper", yref="paper",
            font=dict(size=16, color='red')
        ).update_layout(
            height=450,
            plot_bgcolor='white',
            paper_bgcolor=COLORS['white']
        )
    
# ============================================================================
# PREPARAÇÃO DE DADOS PARA PAINEL GERENCIAL
# ============================================================================

def prepare_manager_data(df):
    """Prepara dados específicos para análise gerencial"""
    
    print("🏢 Preparando dados para análise gerencial...")
    
    df_manager = df.copy()
    
    # 1. ADR por Tipo de Quarto
    if 'reserved_room_type' not in df_manager.columns:
        print("   • Criando tipos de quarto...")
        room_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        weights = [0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.03, 0.02]
        df_manager['reserved_room_type'] = np.random.choice(room_types, len(df), p=weights)
        df_manager['assigned_room_type'] = df_manager['reserved_room_type'].copy()
        
        # Simular algumas mudanças de quarto
        change_mask = np.random.random(len(df)) < 0.15  # 15% mudam de quarto
        df_manager.loc[change_mask, 'assigned_room_type'] = np.random.choice(room_types, change_mask.sum())
    
    # 2. Dados de Ocupação (simulados baseados em padrões reais)
    if 'room_capacity' not in df_manager.columns:
        print("   • Criando dados de ocupação...")
        # Capacidade por tipo de quarto
        room_capacity = {'A': 150, 'B': 120, 'C': 100, 'D': 80, 'E': 60, 'F': 40, 'G': 30, 'H': 20}
        df_manager['room_capacity'] = df_manager['reserved_room_type'].map(room_capacity)
    
    # 3. Estacionamento
    if 'required_car_parking_spaces' not in df_manager.columns:
        print("   • Criando dados de estacionamento...")
        # Probabilidade de solicitar estacionamento varia por perfil
        parking_prob = np.where(df_manager['customer_type'] == 'Corporate', 0.4, 0.2)
        df_manager['required_car_parking_spaces'] = np.random.binomial(1, parking_prob)
    
    # 4. Mudanças de Reserva (Remarcações)
    if 'booking_changes' not in df_manager.columns:
        print("   • Criando dados de remarcações...")
        # Mais mudanças para lead_time alto e reservas corporativas
        change_prob = np.minimum(0.5, df_manager['lead_time'] / 365 * 0.3)
        change_prob = np.where(df_manager['customer_type'] == 'Corporate', change_prob * 1.5, change_prob)
        df_manager['booking_changes'] = np.random.poisson(change_prob)
        df_manager['booking_changes'] = np.minimum(df_manager['booking_changes'], 5)  # Max 5 mudanças
    
    # 5. Criar métricas derivadas
    df_manager['has_parking_request'] = (df_manager['required_car_parking_spaces'] > 0).astype(int)
    df_manager['has_room_change'] = (df_manager['reserved_room_type'] != df_manager['assigned_room_type']).astype(int)
    df_manager['has_booking_changes'] = (df_manager['booking_changes'] > 0).astype(int)
    
    print("✅ Dados gerenciais preparados!")
    return df_manager

# Preparar dados gerenciais
df_manager = prepare_manager_data(df)

# ============================================================================
# COMPONENTES DE FILTROS
# ============================================================================

def create_compact_filters_section():
    """Cria seção de filtros compacta e horizontal"""
    
    try:
        # Obter valores únicos para os dropdowns
        countries = sorted([c for c in df['country'].unique() if pd.notna(c) and str(c) != 'nan'])
        top_countries = df['country'].value_counts().head(10).index.tolist()
        market_segments = sorted([ms for ms in df['market_segment'].unique() if pd.notna(ms)])
        hotels = sorted([h for h in df['hotel'].unique() if pd.notna(h)])
        customer_types = sorted([ct for ct in df['customer_type'].unique() if pd.notna(ct)])
        
        # Ranges para sliders
        adr_min, adr_max = float(df['adr'].min()), float(min(df['adr'].max(), 1000))
        lead_min, lead_max = float(df['lead_time'].min()), float(min(df['lead_time'].max(), 365))
        nights_min, nights_max = float(df['total_nights'].min()), float(min(df['total_nights'].max(), 30))
        guests_min, guests_max = float(df['total_guests'].min()), float(min(df['total_guests'].max(), 10))
        
    except Exception as e:
        print(f"⚠️ Erro ao criar seção de filtros: {e}")
        # Valores padrão seguros
        countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU']
        top_countries = countries
        market_segments = ['Online TA', 'Offline TA/TO', 'Groups']
        hotels = ['City Hotel', 'Resort Hotel']
        customer_types = ['Transient', 'Contract']
        adr_min, adr_max = 0.0, 500.0
        lead_min, lead_max = 0.0, 365.0
        nights_min, nights_max = 1.0, 30.0
        guests_min, guests_max = 1.0, 10.0
    
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5([
                    "🎛️ Filtros Globais"
                ], style={'color': COLORS['white'], 'margin': 0, 'display': 'inline-block', 'fontSize': '18px'}),
                dbc.Badge("Análise Multidimensional", 
                         color= COLORS['accent'], 
                         className="ms-3",
                         style={'fontSize': '11px'}),
                # Status e botões na mesma linha do header
                html.Div([
                    html.Div(id='filter-status', children=[
                        dbc.Badge("🟢 Prontos", color="success", className="me-2", style={'fontSize': '11px'}),
                        html.Small(f"{len(df):,} registros", style={'color': COLORS['white'], 'fontSize': '11px'})
                    ], style={'display': 'inline-block', 'marginRight': '15px'}),
                    
                    dbc.Button("🎯 Aplicar", id='apply-filters-btn', color="light", size="sm", 
                              className="me-2", style={'fontSize': '12px', 'padding': '4px 12px'}),
                    dbc.Button("🔄", id='reset-filters-btn', color="secondary", size="sm", outline=True,
                              style={'fontSize': '12px', 'padding': '4px 8px'})
                ], style={'float': 'right'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'width': '100%'})
        ], style={
            'backgroundColor': COLORS['primary'],
            'borderRadius': '10px 10px 0 0',
            'padding': '10px 20px',
            'minHeight': '50px'
        }),
        
        dbc.CardBody([
            # Linha única com todos os filtros
            dbc.Row([
                # Coluna 1: Filtros Categóricos
                dbc.Col([
                    html.Label("🏨 Hotel", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Dropdown(
                        id='filter-hotel',
                        options=[{'label': h, 'value': h} for h in hotels],
                        value=hotels,
                        multi=True,
                        placeholder="Selecione...",
                        style={'fontSize': '12px'}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("🌍 País", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Dropdown(
                        id='filter-country',
                        options=[{'label': c, 'value': c} for c in countries],
                        value=top_countries,
                        multi=True,
                        placeholder="Selecione...",
                        style={'fontSize': '12px'}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("💼 Segmento", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Dropdown(
                        id='filter-market-segment',
                        options=[{'label': ms, 'value': ms} for ms in market_segments],
                        value=market_segments,
                        multi=True,
                        placeholder="Selecione...",
                        style={'fontSize': '12px'}
                    )
                ], width=2),
                
                # Coluna 2: Sliders Compactos
                dbc.Col([
                    html.Label("💰 ADR", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.RangeSlider(
                        id='filter-adr',
                        min=adr_min,
                        max=adr_max,
                        step=10,
                        value=[adr_min, min(500, adr_max)],
                        marks={
                            int(adr_min): f'${int(adr_min)}',
                            min(500, int(adr_max)): f'${min(500, int(adr_max))}'
                        },
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], width=2),
                
                dbc.Col([
                    html.Label("📅 Lead Time", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.RangeSlider(
                        id='filter-lead-time',
                        min=lead_min,
                        max=min(365, lead_max),
                        step=5,
                        value=[lead_min, min(365, lead_max)],
                        marks={
                            0: '0',
                            min(365, int(lead_max)): f'{min(365, int(lead_max))}d'
                        },
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], width=2),
                
                # Coluna 3: Controles Finais
                dbc.Col([
                    html.Label("📊 Status", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Checklist(
                        id='filter-canceled',
                        options=[
                            {'label': ' ✅ Mantidas', 'value': 0},
                            {'label': ' ❌ Canceladas', 'value': 1}
                        ],
                        value=[0, 1],
                        inline=True,
                        style={'fontSize': '11px'}
                    )
                ], width=2)
            ], className="align-items-end"),
            
            # Linha adicional para sliders extras (colapsível)
            dbc.Collapse([
                html.Hr(style={'margin': '15px 0 10px 0'}),
                dbc.Row([
                    dbc.Col([
                        html.Label("🛏️ Noites", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                        dcc.RangeSlider(
                            id='filter-nights',
                            min=nights_min,
                            max=min(30, nights_max),
                            step=1,
                            value=[nights_min, min(14, nights_max)],
                            marks={1: '1', min(30, int(nights_max)): f'{min(30, int(nights_max))}'},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("👥 Hóspedes", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                        dcc.RangeSlider(
                            id='filter-guests',
                            min=guests_min,
                            max=min(10, guests_max),
                            step=1,
                            value=[guests_min, min(6, guests_max)],
                            marks={1: '1', min(10, int(guests_max)): f'{min(10, int(guests_max))}'},
                            tooltip={"placement": "bottom", "always_visible": False}
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("👤 Tipo Cliente", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                        dcc.Dropdown(
                            id='filter-customer-type',
                            options=[{'label': ct, 'value': ct} for ct in customer_types],
                            value=customer_types,
                            multi=True,
                            placeholder="Selecione...",
                            style={'fontSize': '12px'}
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Label("👨‍👩‍👧‍👦 Família", style={'fontSize': '12px', 'fontWeight': 'bold', 'marginBottom': '4px'}),
                        dcc.Checklist(
                            id='filter-family',
                            options=[
                                {'label': ' 👤 Individual', 'value': 0},
                                {'label': ' 👨‍👩‍👧‍👦 Família', 'value': 1}
                            ],
                            value=[0, 1],
                            inline=True,
                            style={'fontSize': '11px'}
                        )
                    ], width=3)
                ])
            ], id="advanced-filters", is_open=False),
            
            # Botão para expandir filtros avançados
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-chevron-down me-1"), "Filtros Avançados"],
                    id="toggle-advanced-filters",
                    color="link",
                    size="sm",
                    style={'fontSize': '12px', 'padding': '5px 0', 'textDecoration': 'none'}
                )
            ], style={'textAlign': 'center', 'marginTop': '10px'})
            
        ], style={
            'backgroundColor': COLORS['white'], 
            'padding': '15px 20px',
            'borderRadius': '0 0 10px 10px'
        })
    ], style={
        'borderRadius': '10px', 
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'border': f'1px solid {COLORS["primary"]}40',
        'marginBottom': '20px'
    })

# ============================================================================
# FUNÇÃO DE FILTRAGEM DOS DADOS
# ============================================================================

def filter_data(df, hotels, countries, market_segments, customer_types, 
                adr_range, lead_range, nights_range, guests_range, 
                canceled_status, family_status):
    """Aplica todos os filtros nos dados"""
    
    try:
        df_filtered = df.copy()
        
        # Filtros categóricos
        if hotels and len(hotels) > 0:
            df_filtered = df_filtered[df_filtered['hotel'].isin(hotels)]
        
        if countries and len(countries) > 0:
            df_filtered = df_filtered[df_filtered['country'].isin(countries)]
        
        if market_segments and len(market_segments) > 0:
            df_filtered = df_filtered[df_filtered['market_segment'].isin(market_segments)]
        
        if customer_types and len(customer_types) > 0:
            df_filtered = df_filtered[df_filtered['customer_type'].isin(customer_types)]
        
        # Filtros numéricos
        if adr_range:
            df_filtered = df_filtered[
                (df_filtered['adr'] >= adr_range[0]) & 
                (df_filtered['adr'] <= adr_range[1])
            ]
        
        if lead_range:
            df_filtered = df_filtered[
                (df_filtered['lead_time'] >= lead_range[0]) & 
                (df_filtered['lead_time'] <= lead_range[1])
            ]
        
        if nights_range:
            df_filtered = df_filtered[
                (df_filtered['total_nights'] >= nights_range[0]) & 
                (df_filtered['total_nights'] <= nights_range[1])
            ]
        
        if guests_range:
            df_filtered = df_filtered[
                (df_filtered['total_guests'] >= guests_range[0]) & 
                (df_filtered['total_guests'] <= guests_range[1])
            ]
        
        # Filtros especiais
        if canceled_status and len(canceled_status) > 0:
            if 'is_canceled' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['is_canceled'].isin(canceled_status)]
            
        if family_status and len(family_status) > 0:
            if 'is_family' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['is_family'].isin(family_status)]
            else:
                print("⚠️ Coluna 'is_family' não encontrada - ignorando filtro familiar")
        
        return df_filtered
        
    except Exception as e:
        print(f"❌ Erro na filtragem: {str(e)}")
        print("🔄 Retornando dataset original...")
        return df

# ============================================================================
# 5. DASHBOARD INTERATIVO
# ============================================================================

print("\n" + "=" * 80)
print("📊 CRIANDO DASHBOARD INTERATIVO")
print("=" * 80)

# Preparar dados para o dashboard
model_results_df = pd.DataFrame([
    {
        'Modelo': name,
        'Acurácia': results['accuracy'],
        'F1-Score': results['f1_score'],
        'AUC': results['auc']
    }
    for name, results in ml_results.items()
])

ml_results_parts = ml_results['LogisticRegression']['accuracy'] * 100
# Criar aplicação Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Hotel Booking Analysis Dashboard"

# Layout do dashboard
app.layout = dbc.Container([
    # Cabeçalho
    dbc.Row([
        dbc.Col([
            html.H1(
                "🏨 Hotel Booking Analysis Dashboard",
                className="text-center mb-4",
                style={
                    'color': COLORS['dark'],
                    'fontWeight': 'bold',
                    'padding': '20px 0'
                }
            )
        ], width=12)
    ], style={
        'background': (
            f'linear-gradient(135deg, {COLORS["white"]} 0%, '
            f'{COLORS["background"]} 100%)'
        )
    }),

    # ✅ SEÇÃO DE FILTROS GLOBAL (FORA DAS ABAS)
    dbc.Row([
        dbc.Col([
            create_compact_filters_section()
        ], width=12)
    ], className="mb-4"),

    # Abas principais
    dcc.Tabs(id="main-tabs", style={'marginTop': '20px'}, children=[

        # Tab 1: Painel do CEO (SEM FILTROS INTERNOS)
        dcc.Tab(
            label='📈 Painel do CEO',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[
                # Conteúdo será gerado dinamicamente pelo callback
                html.Div(id='overview-content')
            ]
        ),

        # Tab 2: Painel do Gerente
        dcc.Tab(
            label='🏢 Painel do Gerente',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[
                # Conteúdo será gerado dinamicamente pelo callback
                html.Div(id='manager-content')
            ]
        ),

        # Tab 3: Previsão de Cancelamentos 
        dcc.Tab(
            label='🎯 Previsão de Cancelamentos',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[
                # Conteúdo será gerado dinamicamente pelo callback
                html.Div(id='ml-content')
            ]
        ),

        # Tab 4: Simulação 
        dcc.Tab(
            label='🎲 Simulador de Cancelamento',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[
                # Banner explicativo - VERSÃO COMPACTA
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H5("🎲 Simulador de Risco", 
                                   className="alert-heading mb-2",
                                   style={'fontSize': '18px', 'fontWeight': 'bold'}),
                            html.P("Simule cenários e avalie riscos de cancelamento.", 
                                  className="mb-1", style={'fontSize': '14px'}),
                            html.Small("Ferramenta para análise prévia de reservas", 
                                      style={'opacity': '0.8', 'fontSize': '12px'})
                        ], color="info", style={
                            'borderRadius': '8px',
                            'backgroundColor': f'{COLORS["primary"]}10',
                            'border': f'1px solid {COLORS["primary"]}40',
                            'padding': '15px',
                            'marginBottom': '20px'
                        })
                    ], width=12)
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📋 Dados da Reserva",
                                           style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'],
                                                  'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div([
                                    html.H6("⏰ Quando?", style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Antecedência (dias antes do check-in):",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Input(
                                                id='lead-time', type='number', value=50,
                                                className="form-control",
                                                placeholder="Ex: 30 dias"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Duração da estadia (noites):",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Input(
                                                id='total-nights', type='number', value=3,
                                                className="form-control",
                                                placeholder="Ex: 5 noites"
                                            )
                                        ], width=6)
                                    ], className="mb-3"),

                                    html.Hr(),
                                    html.H6("🏨 Onde e Quanto?", style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Tipo de Hotel:",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Dropdown(
                                                id='pred-hotel',
                                                options=[{'label': hotel, 'value': hotel} for hotel in
                                                         df['hotel'].unique()],
                                                value='City Hotel',
                                                className="mb-2"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Valor da diária (R$):",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Input(
                                                id='adr', type='number', value=100,
                                                className="form-control",
                                                placeholder="Ex: 250"
                                            )
                                        ], width=6)
                                    ], className="mb-3"),

                                    html.Hr(),
                                    html.H6("👥 Quem?", style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Perfil do Cliente:",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Dropdown(
                                                id='customer-type',
                                                options=[{'label': ct, 'value': ct} for ct in df['customer_type'].unique()],
                                                value='Transient',
                                                className="mb-2"
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            html.Label("Número de Hóspedes:",
                                                       style={'color': COLORS['dark'], 'fontSize': '14px'}),
                                            dcc.Input(
                                                id='total-guests', type='number', value=2,
                                                className="form-control",
                                                placeholder="Ex: 2 pessoas"
                                            )
                                        ], width=6)
                                    ], className="mb-3"),

                                    html.Br(),
                                    dbc.Button("🎯 Simular Risco de Cancelamento", id='predict-btn',
                                               style={'backgroundColor': COLORS['accent'], 'border': 'none',
                                                      'fontWeight': 'bold', 'fontSize': '16px'},
                                               className="w-100")
                                ])
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={'borderRadius': '12px'})
                    ], width=5),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("📊 Resultado da Simulação",
                                           style={'backgroundColor': COLORS['primary'],
                                                  'color': COLORS['white'],
                                                  'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div(id='prediction-result',
                                         className="text-center",
                                         style={'fontSize': '20px',
                                                'fontWeight': 'bold',
                                                'color': COLORS['dark'],
                                                'padding': '20px'}),
                                html.Br(),
                                dcc.Graph(id='probability-chart')
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={'borderRadius': '12px'})
                    ], width=7)
                ])
            ]
        )
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# ============================================================================
# 6. CALLBACKS DO DASHBOARD
# ============================================================================
@app.callback(
    [Output('prediction-result', 'children'),
     Output('probability-chart', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [dash.dependencies.State('lead-time', 'value'),
     dash.dependencies.State('adr', 'value'),
     dash.dependencies.State('pred-hotel', 'value'),
     dash.dependencies.State('customer-type', 'value'),
     dash.dependencies.State('total-nights', 'value'),
     dash.dependencies.State('total-guests', 'value')]
)
def make_prediction(
    n_clicks, lead_time, adr, hotel, customer_type,
    total_nights, total_guests
):
    if n_clicks is None:
        return html.Div([
            html.P("Preencha os dados da reserva ao lado",
                   style={'fontSize': '18px',
                          'color': COLORS['dark']}),
            html.P(
                "e clique em 'Simular' para ver o risco de cancelamento",
                style={'fontSize': '14px',
                       'color': COLORS['dark'],
                       'opacity': '0.7'}
            )
        ]), go.Figure()

    try:
        # Criar dados de input
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'adr': [adr],
            'customer_type': [customer_type],
            'stays_in_week_nights': [total_nights],
            'stays_in_weekend_nights': [0],
            'adults': [total_guests],
            'children': [0],
            'babies': [0],
            'country': ['PRT'],
            'market_segment': ['Online TA'],
            'distribution_channel': ['TA/TO'],
            'is_repeated_guest': [0],
            'previous_cancellations': [0],
            'previous_bookings_not_canceled': [0],
            'reserved_room_type': ['A'],
            'assigned_room_type': ['A'],
            'booking_changes': [0],
            'deposit_type': ['No Deposit'],
            'agent': [0],
            'company': [0],
            'required_car_parking_spaces': [0],
            'total_of_special_requests': [0],
            'arrival_date_month': ['July'],
            'arrival_date_week_number': [28],
            'arrival_date_day_of_month': [15],
            'total_guests': [total_guests],
            'total_nights': [total_nights],
            'has_special_request': [0],
            'is_family': [1 if total_guests > 1 else 0]
        })

        # Fazer predição
        probability = best_model.predict_proba(input_data)[0]

        cancel_prob = probability[1] * 100
        keep_prob = probability[0] * 100

        # Determinar nível de risco
        if cancel_prob < 30:
            risk_level = "BAIXO"
            risk_color = COLORS['secondary']
            risk_icon = "✅"
            recommendation = (
                "Esta reserva tem baixo risco. Processo normal de "
                "confirmação."
            )
        elif cancel_prob < 60:
            risk_level = "MÉDIO"
            risk_color = COLORS['primary']
            risk_icon = "⚠️"
            recommendation = (
                "Atenção: envie lembretes de confirmação próximo ao check-in."
            )
        else:
            risk_level = "ALTO"
            risk_color = COLORS['accent']
            risk_icon = "🚨"
            recommendation = (
                "Risco elevado! Considere entrar em contato para confirmar "
                "ou oferecer incentivos."
            )

        # Resultado formatado
        result_text = html.Div([
            html.Div([
                html.H3(f"{risk_icon} RISCO {risk_level}",
                        style={'color': risk_color, 'marginBottom': '15px'}),
                html.H1(f"{cancel_prob:.1f}%",
                        style={'fontSize': '3rem',
                               'color': risk_color,
                               'marginBottom': '10px'}),
                html.P("de chance de cancelamento", style={
                    'fontSize': '16px',
                    'color': COLORS['dark'],
                    'opacity': '0.8'
                }),
            ], style={'marginBottom': '20px'}),
            html.Hr(),
            html.Div([
                html.P("💡 Recomendação:",
                       style={'fontWeight': 'bold',
                              'color': COLORS['secondary'],
                              'marginBottom': '10px'}),
                html.P(recommendation,
                       style={'fontSize': '14px',
                              'color': COLORS['dark']})
            ])
        ])

        # Gráfico de probabilidade
        prob_fig = go.Figure()

        prob_fig.add_trace(go.Bar(
            x=['Manterá a Reserva', 'Cancelará'],
            y=[keep_prob, cancel_prob],
            marker_color=[COLORS['secondary'], COLORS['accent']],
            text=[f'{keep_prob:.1f}%', f'{cancel_prob:.1f}%'],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black')
        ))

        prob_fig.update_layout(
            title='Probabilidade de Manutenção vs Cancelamento',
            plot_bgcolor='white',
            paper_bgcolor=COLORS['white'],
            font_color=COLORS['dark'],
            title_font_color=COLORS['dark'],
            yaxis_title="Probabilidade (%)",
            xaxis_title="",
            showlegend=False,
            height=350
        )

        return result_text, prob_fig

    except Exception as e:
        error_msg = html.Div([
            html.H4("⚠️ Erro na Simulação", style={'color': COLORS['accent']}),
            html.P(f"Detalhes: {str(e)}",
                   style={'fontSize': '14px',
                          'color': COLORS['dark']})
        ])
        return error_msg, go.Figure()

# ============================================================================
# CALLBACKS DOS FILTROS INTERATIVOS
# ============================================================================


# Callback para toggle dos filtros avançados
@app.callback(
    [Output("advanced-filters", "is_open"),
     Output("toggle-advanced-filters", "children")],
    [Input("toggle-advanced-filters", "n_clicks")],
    [dash.dependencies.State("advanced-filters", "is_open")]
)
def toggle_advanced_filters(n_clicks, is_open):
    if n_clicks:
        if is_open:
            return False, [html.I(className="fas fa-chevron-down me-1"), "Filtros Avançados"]
        else:
            return True, [html.I(className="fas fa-chevron-up me-1"), "Ocultar Filtros Avançados"]
    
    return False, [html.I(className="fas fa-chevron-down me-1"), "Filtros Avançados"]


# Callback para resetar filtros 
@app.callback(
    [Output('filter-hotel', 'value'),
     Output('filter-country', 'value'),
     Output('filter-market-segment', 'value'),
     Output('filter-customer-type', 'value'),
     Output('filter-adr', 'value'),
     Output('filter-lead-time', 'value'),
     Output('filter-nights', 'value'),
     Output('filter-guests', 'value'),
     Output('filter-canceled', 'value'),
     Output('filter-family', 'value')],
    [Input('reset-filters-btn', 'n_clicks')],
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    if not n_clicks or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    
    print(f"🔄 Resetando filtros (cliques: {n_clicks})")
    
    # ✅ CORREÇÃO: Valores padrão seguros
    try:
        countries = sorted([c for c in df['country'].unique() if pd.notna(c)])
        top_countries = df['country'].value_counts().head(10).index.tolist()
        market_segments = sorted(df['market_segment'].unique())
        hotels = sorted(df['hotel'].unique())
        customer_types = sorted(df['customer_type'].unique())
        
        adr_min, adr_max = float(df['adr'].min()), float(min(df['adr'].max(), 1000))
        lead_min, lead_max = float(df['lead_time'].min()), float(min(df['lead_time'].max(), 365))
        nights_min, nights_max = float(df['total_nights'].min()), float(min(df['total_nights'].max(), 30))
        guests_min, guests_max = float(df['total_guests'].min()), float(min(df['total_guests'].max(), 10))
        
    except Exception as e:
        print(f"⚠️ Erro ao calcular valores padrão: {e}")
        # Valores fallback
        countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU']
        top_countries = countries[:10]
        market_segments = ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate']
        hotels = ['City Hotel', 'Resort Hotel']
        customer_types = ['Transient', 'Contract', 'Transient-Party', 'Group']
        adr_min, adr_max = 0.0, 500.0
        lead_min, lead_max = 0.0, 365.0
        nights_min, nights_max = 1.0, 30.0
        guests_min, guests_max = 1.0, 10.0
    
    return (
        hotels,  # filter-hotel
        top_countries,  # filter-country
        market_segments,  # filter-market-segment
        customer_types,  # filter-customer-type
        [adr_min, min(500.0, adr_max)],  # filter-adr
        [lead_min, min(365.0, lead_max)],  # filter-lead-time
        [nights_min, min(14.0, nights_max)],  # filter-nights
        [guests_min, min(6.0, guests_max)],  # filter-guests
        [0, 1],  # filter-canceled
        [0, 1]   # filter-family
    )


# Callback separado para status dos filtros
@app.callback(
    Output('filter-status', 'children'),
    [Input('apply-filters-btn', 'n_clicks'),
     Input('reset-filters-btn', 'n_clicks')],
    [dash.dependencies.State('filter-hotel', 'value'),
     dash.dependencies.State('filter-country', 'value'),
     dash.dependencies.State('filter-market-segment', 'value'),
     dash.dependencies.State('filter-customer-type', 'value'),
     dash.dependencies.State('filter-adr', 'value'),
     dash.dependencies.State('filter-lead-time', 'value'),
     dash.dependencies.State('filter-nights', 'value'),
     dash.dependencies.State('filter-guests', 'value'),
     dash.dependencies.State('filter-canceled', 'value'),
     dash.dependencies.State('filter-family', 'value')]
)
def update_filter_status(apply_clicks, reset_clicks, hotels, countries, market_segments, 
                        customer_types, adr_range, lead_range, nights_range, 
                        guests_range, canceled_status, family_status):
    
    # Determinar qual botão foi clicado
    ctx = callback_context
    if not ctx.triggered:
        # Status inicial
        try:
            top_countries_data = df[df['country'].isin(df['country'].value_counts().head(10).index)]
            estimated_records = len(top_countries_data)
        except Exception as e:
            estimated_records = len(df)
            
        return [
            dbc.Badge("🟢 Prontos", color="success", className="me-1", style={'fontSize': '11px'}),
            html.Small(f"{estimated_records:,} reg.", style={'color': COLORS['white'], 'fontSize': '11px'})
        ]
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'reset-filters-btn' and reset_clicks:
        try:
            top_countries_data = df[df['country'].isin(df['country'].value_counts().head(10).index)]
            records_count = len(top_countries_data)
        except Exception as e:
            records_count = len(df)
            
        return [
            dbc.Badge("🔄 Reset", color="info", className="me-1", style={'fontSize': '11px'}),
            html.Small(f"{records_count:,} reg.", style={'color': COLORS['white'], 'fontSize': '11px'})
        ]
    
    # Aplicar filtros para calcular status
    try:
        df_filtered = filter_data(df, hotels, countries, market_segments, customer_types,
                                 adr_range, lead_range, nights_range, guests_range,
                                 canceled_status, family_status)
        
        total_bookings = len(df_filtered)
        
        if total_bookings == 0:
            status_color = "danger"
            status_text = "❌ Vazio"
        elif total_bookings < len(df) * 0.05:
            status_color = "warning" 
            status_text = "⚠️ Poucos"
        else:
            status_color = "success"
            status_text = "✅ OK"
        
        return [
            dbc.Badge(status_text, color=status_color, className="me-1", style={'fontSize': '11px'}),
            html.Small(f"{total_bookings:,} reg.", style={'color': COLORS['white'], 'fontSize': '11px'})
        ]
        
    except Exception as e:
        return [
            dbc.Badge("⚠️ Erro", color="danger", className="me-1", style={'fontSize': '11px'}),
            html.Small("N/A", style={'color': COLORS['white'], 'fontSize': '11px'})
        ]


# Callback principal para atualizar conteúdo
@app.callback(
    [Output('overview-content', 'children'),
     Output('manager-content', 'children'),
     Output('ml-content', 'children')],
    [Input('apply-filters-btn', 'n_clicks'),
     Input('main-tabs', 'active_tab')],  # ← ADICIONAR: Trigger para carregar abas
    [dash.dependencies.State('filter-hotel', 'value'),
     dash.dependencies.State('filter-country', 'value'),
     dash.dependencies.State('filter-market-segment', 'value'),
     dash.dependencies.State('filter-customer-type', 'value'),
     dash.dependencies.State('filter-adr', 'value'),
     dash.dependencies.State('filter-lead-time', 'value'),
     dash.dependencies.State('filter-nights', 'value'),
     dash.dependencies.State('filter-guests', 'value'),
     dash.dependencies.State('filter-canceled', 'value'),
     dash.dependencies.State('filter-family', 'value')]
)
def update_dashboard_content(apply_clicks, active_tab, hotels, countries, market_segments, 
                           customer_types, adr_range, lead_range, nights_range, 
                           guests_range, canceled_status, family_status):
    
    print(f"🎯 Atualizando dashboard - Cliques: {apply_clicks}, Aba: {active_tab}")  # Debug
    
    # ✅ USAR VALORES PADRÃO SE NENHUM FILTRO APLICADO
    if not apply_clicks or apply_clicks == 0:
        # Primeira carga - usar todos os dados
        df_filtered = df
        df_manager_filtered = df_manager
        print("📊 Primeira carga - usando todos os dados")
    else:
        # Aplicar filtros
        df_filtered = filter_data(df, hotels, countries, market_segments, customer_types,
                                 adr_range, lead_range, nights_range, guests_range,
                                 canceled_status, family_status)
        df_manager_filtered = filter_data(df_manager, hotels, countries, market_segments, customer_types,
                                         adr_range, lead_range, nights_range, guests_range,
                                         canceled_status, family_status)
        print(f"🔍 Filtros aplicados - {len(df_filtered):,} registros")
    
    # Verificar se há dados
    if len(df_filtered) == 0:
        empty_content = dbc.Alert([
            html.H4("🔍 Nenhum resultado encontrado", className="alert-heading"),
            html.P("Tente ajustar os filtros para obter resultados. Use o botão 'Resetar' para voltar aos valores padrão."),
        ], color="warning")
        
        return empty_content, empty_content, empty_content
    
    # Calcular métricas filtradas
    total_bookings = len(df_filtered)
    cancel_rate = df_filtered['is_canceled'].mean() * 100 if len(df_filtered) > 0 else 0
    avg_adr = df_filtered['adr'].mean() if len(df_filtered) > 0 else 0
    avg_lead = df_filtered['lead_time'].mean() if len(df_filtered) > 0 else 0
    
    if total_bookings == 0:
        # Conteúdo vazio
        empty_content = dbc.Alert([
            html.H4("🔍 Nenhum resultado encontrado", className="alert-heading"),
            html.P("Tente ajustar os filtros para obter resultados. Use o botão 'Resetar' para voltar aos valores padrão."),
        ], color="warning")
        
        return empty_content, empty_content

    # ========== CONTEÚDO DA ABA VISÃO do CEO ==========
    overview_content = html.Div([
        # Banner explicativo - VERSÃO COMPACTA
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📊 Visão Geral do Negócio", 
                           className="alert-heading mb-2", 
                           style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"Análise de {total_bookings:,} reservas para decisões estratégicas.", 
                          className="mb-1", style={'fontSize': '14px'}),
                    html.Small(f"Status: {'Filtrado' if apply_clicks else 'Completo'}", 
                              style={'opacity': '0.8', 'fontSize': '12px'})
                ], color="primary", style={
                    'borderRadius': '8px',
                    'backgroundColor': f'{COLORS["primary"]}10',
                    'border': f'1px solid {COLORS["primary"]}40',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
            ], width=12)
        ], className="mb-3"),

        # Cards de métricas principais
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("🏨", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{total_bookings:,}", 
                                   style={'color': COLORS['primary'], 'fontWeight': 'bold', 
                                          'fontSize': '2.5rem', 'marginBottom': '5px'}),
                            html.P("Total de Reservas", 
                                  style={'color': COLORS['dark'], 'marginBottom': '5px', 'fontSize': '16px'}),
                            html.Small(f"{'Filtrado' if apply_clicks else 'Completo'}", 
                                      style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'border': 'none'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("⚠️", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{cancel_rate:.1f}%", 
                                   style={'color': COLORS['accent'], 'fontWeight': 'bold', 
                                          'fontSize': '2.5rem', 'marginBottom': '5px'}),
                            html.P("Taxa de Cancelamento", 
                                  style={'color': COLORS['dark'], 'marginBottom': '5px', 'fontSize': '16px'}),
                            html.Small("Média atual", 
                                      style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'border': 'none'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("💰", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"${avg_adr:.0f}", 
                                   style={'color': COLORS['secondary'], 'fontWeight': 'bold', 
                                          'fontSize': '2.5rem', 'marginBottom': '5px'}),
                            html.P("Diária Média (ADR)", 
                                  style={'color': COLORS['dark'], 'marginBottom': '5px', 'fontSize': '16px'}),
                            html.Small("Receita média por quarto", 
                                      style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'border': 'none'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("📅", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{avg_lead:.0f}", 
                                   style={'color': COLORS['primary'], 'fontWeight': 'bold', 
                                          'fontSize': '2.5rem', 'marginBottom': '5px'}),
                            html.P("Antecedência Média", 
                                  style={'color': COLORS['dark'], 'marginBottom': '5px', 'fontSize': '16px'}),
                            html.Small("Dias antes do check-in", 
                                      style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'border': 'none'})
            ], width=3)
        ], className="mb-4"),

        # ========== LINHA 1: Performance Hoteleira e Segmentos ==========
        dbc.Row([
            # Performance por tipo de hotel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(f"🏨 Performance por Tipo de Hotel",
                                   style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 
                                          'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.CardBody([
                        create_hotel_performance_chart(df_filtered)
                    ], style={'backgroundColor': COLORS['white']})
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6),
            
            # Análise por Segmento de Mercado
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("💼 Análise por Segmento de Mercado",
                                   style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 
                                          'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.CardBody([
                        create_market_segment_chart(df_filtered)
                    ], style={'backgroundColor': COLORS['white']})
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6)
        ], className="mb-4"),

        # ========== LINHA 2: Análise Geográfica e Sazonalidade ==========
        dbc.Row([
            # Análise por Países (Top 10)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌍 Principais Países de Origem",
                                   style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 
                                          'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.CardBody([
                        create_countries_chart(df_filtered)
                    ], style={'backgroundColor': COLORS['white']})
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6),
            
            # Sazonalidade Mensal
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📅 Sazonalidade das Reservas",
                                   style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 
                                          'fontWeight': 'bold', 'fontSize': '16px'}),
                    dbc.CardBody([
                        create_monthly_chart(df_filtered)
                    ], style={'backgroundColor': COLORS['white']})
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6)
        ], className="mb-4")
    ])

    # ========== CONTEÚDO DA ABA ML ==========  

    ml_content = create_ml_dashboard_optimized(df_filtered)
    
    # ========== CONTEÚDO DA ABA PAINEL DO GERENTE ==========
    manager_content = create_manager_dashboard(df_manager_filtered)

    return overview_content, manager_content, ml_content

# ============================================================================
# FUNÇÕES AUXILIARES PARA GRÁFICOS 
# ============================================================================


def create_manager_dashboard(df_manager_filtered):
    """Cria o dashboard completo do gerente"""
    
    total_bookings = len(df_manager_filtered)
    
    if total_bookings == 0:
        return dbc.Alert([
            html.H4("🔍 Nenhum dado disponível", className="alert-heading"),
            html.P("Ajuste os filtros para visualizar as análises gerenciais."),
        ], color="warning")
    
    # Calcular métricas principais
    avg_adr = df_manager_filtered['adr'].mean()
    total_remarcacoes = df_manager_filtered['booking_changes'].sum()
    taxa_estacionamento = df_manager_filtered['has_parking_request'].mean() * 100
    taxa_mudanca_quarto = df_manager_filtered['has_room_change'].mean() * 100

    return html.Div([
        # Banner explicativo
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("🏢 Painel Operacional do Gerente", 
                           className="alert-heading mb-2",
                           style={'fontSize': '18px', 'fontWeight': 'bold'}),
                    html.P(f"Análise operacional de {total_bookings:,} reservas.", 
                          className="mb-1", style={'fontSize': '14px'}),
                    html.Small("Métricas atualizadas", 
                              style={'opacity': '0.8', 'fontSize': '12px'})
                ], color="info", style={
                    'borderRadius': '8px',
                    'backgroundColor': f'{COLORS["secondary"]}10',
                    'border': f'1px solid {COLORS["secondary"]}40',
                    'padding': '15px',
                    'marginBottom': '20px'
                })
            ], width=12)
        ], className="mb-3"),

        # Cards de métricas principais
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("💰", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"${avg_adr:.0f}", 
                                   style={'color': COLORS['secondary'], 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                            html.P("ADR Médio Atual", style={'color': COLORS['dark'], 'fontSize': '16px', 'marginBottom': '5px'}),
                            html.Small("Average Daily Rate", style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("🔄", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{total_remarcacoes:,}", 
                                   style={'color': COLORS['primary'], 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                            html.P("Remarcações", style={'color': COLORS['dark'], 'fontSize': '16px', 'marginBottom': '5px'}),
                            html.Small("Total de alterações", style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("🚗", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{taxa_estacionamento:.1f}%", 
                                   style={'color': COLORS['accent'], 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                            html.P("Solicitam Estacionamento", style={'color': COLORS['dark'], 'fontSize': '16px', 'marginBottom': '5px'}),
                            html.Small("Taxa de solicitação", style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], width=3),

            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("🏠", style={'fontSize': '2.5rem', 'marginBottom': '10px'}),
                            html.H2(f"{taxa_mudanca_quarto:.1f}%", 
                                   style={'color': COLORS['primary'], 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                            html.P("Mudanças de Quarto", style={'color': COLORS['dark'], 'fontSize': '16px', 'marginBottom': '5px'}),
                            html.Small("Reservado ≠ Atribuído", style={'color': COLORS['dark'], 'opacity': '0.7'})
                        ], style={'textAlign': 'center'})
                    ], style={'padding': '25px'})
                ], style={'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
            ], width=3)
        ], className="mb-4"),

        # Linha 1: ADR por Tipo de Quarto e Ocupação
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-money-bill-wave me-2"),
                            "💰 ADR por Tipo de Quarto"
                        ], style={'color': COLORS['white'], 'margin': 0})
                    ], style={'backgroundColor': COLORS['secondary'], 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        create_adr_room_chart(df_manager_filtered)
                    ])
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-bed me-2"),
                            "🛏️ Ocupação por Tipo de Quarto"
                        ], style={'color': COLORS['white'], 'margin': 0})
                    ], style={'backgroundColor': COLORS['primary'], 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        create_occupancy_room_chart(df_manager_filtered)
                    ])
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6)
        ], className="mb-4"),

        # Linha 2: Análise de Estacionamento e Remarcações
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-car me-2"),
                            "🚗 Estacionamento vs Cancelamento"
                        ], style={'color': COLORS['white'], 'margin': 0})
                    ], style={'backgroundColor': COLORS['accent'], 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        create_parking_analysis_chart(df_manager_filtered)
                    ])
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-exchange-alt me-2"),
                            "🔄 Análise de Remarcações"
                        ], style={'color': COLORS['white'], 'margin': 0})
                    ], style={'backgroundColor': COLORS['primary'], 'fontWeight': 'bold'}),
                    dbc.CardBody([
                        create_booking_changes_chart(df_manager_filtered)
                    ])
                ], style={'borderRadius': '12px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], width=6)
        ], className="mb-4"),
    ])


def create_hotel_performance_chart(df_filtered):
    """Cria gráfico de performance hoteleira"""
    
    if len(df_filtered) == 0:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="Nenhum dado disponível", 
                x=0.5, y=0.5, xref="paper", yref="paper"
            )
        )
    
    try:
        return dcc.Graph(
            figure=make_subplots(
                rows=1, cols=2,
                subplot_titles=('Volume de Reservas', 'Taxa de Cancelamento'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            ).add_trace(
                go.Bar(
                    x=df_filtered['hotel'].value_counts().index,
                    y=df_filtered['hotel'].value_counts().values,
                    marker_color=[COLORS['primary'], COLORS['secondary']],
                    text=df_filtered['hotel'].value_counts().values,
                    textposition='auto',
                    texttemplate='%{text:,}',
                    hovertemplate='<b>%{x}</b><br>Reservas: %{y:,}<extra></extra>',
                    showlegend=False
                ), row=1, col=1
            ).add_trace(
                go.Bar(
                    x=df_filtered.groupby('hotel')['is_canceled'].mean().index,
                    y=(df_filtered.groupby('hotel')['is_canceled'].mean().values * 100),
                    marker_color=[COLORS['accent'], COLORS['primary']],
                    text=[f"{v:.1f}%" for v in (df_filtered.groupby('hotel')['is_canceled'].mean().values * 100)],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Cancelamento: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ), row=1, col=2
            ).update_layout(
                height=400, showlegend=False, plot_bgcolor='white',
                paper_bgcolor=COLORS['white'], font_color=COLORS['dark'],
                margin=dict(l=20, r=20, t=60, b=20)
            )
        )
    except Exception as e:
        return html.P(f"Erro ao criar gráfico: {str(e)}", style={'color': 'red'})


def create_ml_analysis_content(df_filtered):
    """Cria conteúdo da análise ML"""
    
    if len(df_filtered) < 10:
        return html.Div([
            dbc.Alert([
                html.H5("⚠️ Dados Insuficientes", className="alert-heading"),
                html.P("Mínimo de 10 registros necessários para análise ML. Ajuste os filtros."),
            ], color="warning")
        ])
    
    try:
        # Análise simples de perfis
        cluster_summary = df_filtered.groupby(['hotel', 'market_segment']).agg({
            'is_canceled': 'mean',
            'adr': 'mean',
            'lead_time': 'mean'
        }).round(2)
        
        return html.Div([
            html.P(f"Análise de {len(df_filtered):,} registros filtrados:", 
                   style={'marginBottom': '20px'}),
            
            dcc.Graph(
                figure=go.Figure(data=go.Heatmap(
                    z=cluster_summary['is_canceled'].values.reshape(-1, 1),
                    x=['Taxa de Cancelamento'],
                    y=[f"{idx[0]} - {idx[1]}" for idx in cluster_summary.index],
                    colorscale='RdYlBu_r',
                    text=[[f"{v:.1%}"] for v in cluster_summary['is_canceled'].values],
                    texttemplate="%{text}",
                    hovertemplate='<b>%{y}</b><br>Cancelamento: %{z:.1%}<extra></extra>'
                )).update_layout(
                    title="Taxa de Cancelamento por Segmento",
                    height=400,
                    plot_bgcolor='white',
                    paper_bgcolor=COLORS['white']
                )
            )
        ])
        
    except Exception as e:
        return html.P(f"Erro na análise ML: {str(e)}", style={'color': 'red'})


def create_market_segment_chart(df_filtered):
    """Cria gráfico de análise por segmento de mercado"""
    
    if len(df_filtered) == 0:
        return dcc.Graph(figure=go.Figure().add_annotation(
            text="Nenhum dado disponível", x=0.5, y=0.5, xref="paper", yref="paper"
        ))
    
    try:
        # Análise por segmento
        segment_stats = df_filtered.groupby('market_segment').agg({
            'is_canceled': 'mean',
            'adr': 'mean'
        }).round(3)
        
        segment_counts = df_filtered['market_segment'].value_counts()
        
        return dcc.Graph(
            figure=make_subplots(
                rows=2, cols=1,
                subplot_titles=('Volume por Segmento', 'Taxa de Cancelamento por Segmento'),
                vertical_spacing=0.15,
                specs=[[{"type": "bar"}], [{"type": "bar"}]]
            ).add_trace(
                go.Bar(
                    x=segment_counts.index,
                    y=segment_counts.values,
                    marker_color=COLORS['secondary'],
                    text=segment_counts.values,
                    textposition='auto',
                    texttemplate='%{text:,}',
                    hovertemplate='<b>%{x}</b><br>Reservas: %{y:,}<extra></extra>',
                    showlegend=False
                ), row=1, col=1
            ).add_trace(
                go.Bar(
                    x=segment_stats.index,
                    y=segment_stats['is_canceled'] * 100,
                    marker_color=COLORS['accent'],
                    text=[f"{v:.1f}%" for v in segment_stats['is_canceled'] * 100],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Cancelamento: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ), row=2, col=1
            ).update_layout(
                height=500, 
                plot_bgcolor='white',
                paper_bgcolor=COLORS['white'], 
                font_color=COLORS['dark'],
                margin=dict(l=20, r=20, t=60, b=20)
            )
        )
    except Exception as e:
        return html.P(f"Erro ao criar gráfico: {str(e)}", style={'color': 'red'})


def create_countries_chart(df_filtered):
    """Cria gráfico dos principais países"""
    
    if len(df_filtered) == 0:
        return dcc.Graph(figure=go.Figure().add_annotation(
            text="Nenhum dado disponível", x=0.5, y=0.5, xref="paper", yref="paper"
        ))
    
    try:
        # Top 10 países
        country_stats = df_filtered['country'].value_counts().head(10)
        country_cancel = df_filtered.groupby('country')['is_canceled'].mean().loc[country_stats.index]
        
        return dcc.Graph(
            figure=make_subplots(
                rows=2, cols=1,
                subplot_titles=('Top 10 Países por Volume', 'Taxa de Cancelamento por País'),
                vertical_spacing=0.15,
                specs=[[{"type": "bar"}], [{"type": "bar"}]]
            ).add_trace(
                go.Bar(
                    x=country_stats.index,
                    y=country_stats.values,
                    marker_color=COLORS['primary'],
                    text=country_stats.values,
                    textposition='auto',
                    texttemplate='%{text:,}',
                    hovertemplate='<b>%{x}</b><br>Reservas: %{y:,}<extra></extra>',
                    showlegend=False
                ), row=1, col=1
            ).add_trace(
                go.Bar(
                    x=country_cancel.index,
                    y=country_cancel.values * 100,
                    marker_color=COLORS['accent'],
                    text=[f"{v:.1f}%" for v in country_cancel.values * 100],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Cancelamento: %{y:.1f}%<extra></extra>',
                    showlegend=False
                ), row=2, col=1
            ).update_layout(
                height=500, 
                plot_bgcolor='white',
                paper_bgcolor=COLORS['white'], 
                font_color=COLORS['dark'],
                margin=dict(l=20, r=20, t=60, b=20)
            )
        )
    except Exception as e:
        return html.P(f"Erro ao criar gráfico: {str(e)}", style={'color': 'red'})


def create_monthly_chart(df_filtered):
    """Cria gráfico de sazonalidade mensal"""
    
    if len(df_filtered) == 0:
        return dcc.Graph(figure=go.Figure().add_annotation(
            text="Nenhum dado disponível", x=0.5, y=0.5, xref="paper", yref="paper"
        ))
    
    try:
        # Ordem correta dos meses
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        monthly_counts = df_filtered['arrival_date_month'].value_counts().reindex(month_order, fill_value=0)
        monthly_cancel = df_filtered.groupby('arrival_date_month')['is_canceled'].mean().reindex(month_order, fill_value=0)
        
        return dcc.Graph(
            figure=go.Figure().add_trace(
                go.Scatter(
                    x=monthly_counts.index,
                    y=monthly_counts.values,
                    mode='lines+markers',
                    name='Volume de Reservas',
                    line=dict(color=COLORS['primary'], width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>%{x}</b><br>Reservas: %{y:,}<extra></extra>'
                )
            ).add_trace(
                go.Scatter(
                    x=monthly_cancel.index,
                    y=monthly_cancel.values * 100,
                    mode='lines+markers',
                    name='Taxa Cancelamento (%)',
                    line=dict(color=COLORS['accent'], width=3),
                    marker=dict(size=8),
                    yaxis='y2',
                    hovertemplate='<b>%{x}</b><br>Cancelamento: %{y:.1f}%<extra></extra>'
                )
            ).update_layout(
                title="Sazonalidade: Volume vs Taxa de Cancelamento",
                xaxis_title="Mês",
                yaxis=dict(title="Volume de Reservas", side="left"),
                yaxis2=dict(title="Taxa de Cancelamento (%)", side="right", overlaying="y"),
                height=400,
                plot_bgcolor='white',
                paper_bgcolor=COLORS['white'],
                font_color=COLORS['dark'],
                hovermode='x unified',
                legend=dict(x=0.7, y=1, bgcolor='rgba(255,255,255,0.8)')
            )
        )
    except Exception as e:
        return html.P(f"Erro ao criar gráfico: {str(e)}", style={'color': 'red'})


def create_occupancy_room_chart(df):
    """Cria gráfico de ocupação por tipo de quarto - CORRIGIDO"""
    
    # ✅ VERIFICAR SE DADOS EXISTEM
    if len(df) == 0 or 'reserved_room_type' not in df.columns:
        return dcc.Graph(
            figure=go.Figure().add_annotation(
                text="Dados de quartos não disponíveis", 
                x=0.5, y=0.5, xref="paper", yref="paper",
                font=dict(size=16, color=COLORS['dark'])
            ).update_layout(
                height=400,
                plot_bgcolor='white',
                paper_bgcolor=COLORS['white']
            )
        )
    
    try:
        # ✅ CORREÇÃO: Usar nomes corretos das colunas
        # Simular dados de ocupação (baseado em reservas não canceladas)
        occupancy_data = df[df['is_canceled'] == 0].groupby('reserved_room_type').agg({
            'room_capacity': 'first',  # Capacidade do quarto
            'adr': 'count'  # Número de reservas (usar adr como contador)
        }).reset_index()
        
        occupancy_data.columns = ['room_type', 'capacity', 'bookings']
        
        # Verificar se temos dados
        if len(occupancy_data) == 0:
            return dcc.Graph(
                figure=go.Figure().add_annotation(
                    text="Nenhuma reserva confirmada disponível", 
                    x=0.5, y=0.5, xref="paper", yref="paper"
                )
            )
        
        occupancy_data['occupancy_rate'] = (occupancy_data['bookings'] / occupancy_data['capacity'] * 100).round(1)
        occupancy_data = occupancy_data.sort_values('occupancy_rate', ascending=False)
        
        # Análise de receita por quarto
        revenue_by_room = df[df['is_canceled'] == 0].groupby('reserved_room_type').agg({
            'adr': 'sum',  # Receita total
            'total_nights': 'sum'  # Total de noites
        }).reset_index()
        
        # ✅ CORREÇÃO: Renomear colunas corretamente
        revenue_by_room.columns = ['room_type', 'total_revenue', 'total_nights']
        
        # Verificar se total_nights > 0 para evitar divisão por zero
        revenue_by_room = revenue_by_room[revenue_by_room['total_nights'] > 0]
        revenue_by_room['revenue_per_night'] = (revenue_by_room['total_revenue'] / revenue_by_room['total_nights']).round(2)
        
