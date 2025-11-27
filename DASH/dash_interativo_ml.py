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
from dash import Input, Output, dcc, html
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

    print("🎯 Realizando análise de clusters...")

    # Selecionar features numéricas para clustering
    numeric_features = [
        'lead_time', 'adr', 'adults', 'children', 'babies',
        'stays_in_weekend_nights', 'stays_in_week_nights',
        'previous_cancellations', 'booking_changes',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]

    # Garantir que as features existem
    available_numeric = [f for f in numeric_features if f in df.columns]
    X_cluster = df[available_numeric].fillna(0)

    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Calcular silhueta (desabilitado para performance)
    # Pode ser demorado com muitos dados
    # silhouette_avg = silhouette_score(X_scaled, clusters)
    # print(f"   • Score de silhueta: {silhouette_avg:.4f}")
    print(f"   • {len(set(clusters))} clusters identificados")

    # PCA para visualização
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Variância explicada pelos componentes principais
    variance_explained = pca.explained_variance_ratio_
    print(f"   • Variância explicada PC1: {variance_explained[0]:.2%}")
    print(f"   • Variância explicada PC2: {variance_explained[1]:.2%}")
    print(f"   • Variância total explicada: {sum(variance_explained):.2%}")

    # Análise dos clusters
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    cluster_analysis = df_clustered.groupby('cluster').agg({
        'is_canceled': 'mean',
        'adr': 'mean',
        'lead_time': 'mean',
        'total_guests': 'mean',
        'total_nights': 'mean'
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
        'cluster_labels': cluster_labels
    }


# Executar clustering (usando amostra para velocidade)
print(
    "\n🔮 Para clustering, usando amostra de 10.000 registros "
    "(performance)..."
)
df_cluster_sample = df.sample(n=min(10000, len(df)), random_state=42)
clustering_results = perform_clustering(df_cluster_sample)

print("✅ Análise de clusters concluída!")

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

    # Abas principais
    dcc.Tabs(style={'marginTop': '20px'}, children=[

        # Tab 1: Visão Geral
        dcc.Tab(
            label='📈 Painel de Gestão',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[

                # Banner explicativo
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H4(
                                "📊 Visão Geral do Seu Negócio",
                                className="alert-heading"
                            ),
                            html.P(
                                "Acompanhe as principais métricas e "
                                "tendências do seu hotel em tempo real. "
                                "Informações essenciais para tomada de "
                                "decisões estratégicas."
                            ),
                        ], color="primary", style={
                            'borderRadius': '12px',
                            'backgroundColor': f'{COLORS["primary"]}15',
                            'border': f'2px solid {COLORS["primary"]}'
                        })
                    ], width=12)
                ], className="mb-4"),

                # Cards de métricas principais - Layout moderno
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H3(
                                        "🏨",
                                        style={
                                            'fontSize': '2.5rem',
                                            'marginBottom': '10px'
                                        }
                                    ),
                                    html.H2(
                                        f"{eda_results['total_bookings']:,}",
                                        style={
                                            'color': COLORS['primary'],
                                            'fontWeight': 'bold',
                                            'fontSize': '2.5rem',
                                            'marginBottom': '5px'
                                        }
                                    ),
                                    html.P(
                                        "Total de Reservas",
                                        style={
                                            'color': COLORS['dark'],
                                            'marginBottom': '5px',
                                            'fontSize': '16px'
                                        }
                                    ),
                                    html.Small(
                                        "Volume total registrado",
                                        style={
                                            'color': COLORS['dark'],
                                            'opacity': '0.7'
                                        }
                                    )
                                ], style={'textAlign': 'center'})
                            ], style={'padding': '25px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'border': 'none'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H3(
                                        "⚠️",
                                        style={
                                            'fontSize': '2.5rem',
                                            'marginBottom': '10px'
                                        }
                                    ),
                                    html.H2(
                                        f"{eda_results['cancel_rate']:.1f}%",
                                        style={
                                            'color': COLORS['accent'],
                                            'fontWeight': 'bold',
                                            'fontSize': '2.5rem',
                                            'marginBottom': '5px'
                                        }
                                    ),
                                    html.P(
                                        "Taxa de Cancelamento",
                                        style={
                                            'color': COLORS['dark'],
                                            'marginBottom': '5px',
                                            'fontSize': '16px'
                                        }
                                    ),
                                    html.Small(
                                        "Média geral do período",
                                        style={
                                            'color': COLORS['dark'],
                                            'opacity': '0.7'
                                        }
                                    )
                                ], style={'textAlign': 'center'})
                            ], style={'padding': '25px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'border': 'none'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H3(
                                        "💰",
                                        style={
                                            'fontSize': '2.5rem',
                                            'marginBottom': '10px'
                                        }
                                    ),
                                    html.H2(
                                        f"${eda_results['avg_adr']:.0f}",
                                        style={
                                            'color': COLORS['secondary'],
                                            'fontWeight': 'bold',
                                            'fontSize': '2.5rem',
                                            'marginBottom': '5px'
                                        }
                                    ),
                                    html.P(
                                        "Diária Média (ADR)",
                                        style={
                                            'color': COLORS['dark'],
                                            'marginBottom': '5px',
                                            'fontSize': '16px'
                                        }
                                    ),
                                    html.Small(
                                        "Receita média por quarto",
                                        style={
                                            'color': COLORS['dark'],
                                            'opacity': '0.7'
                                        }
                                    )
                                ], style={'textAlign': 'center'})
                            ], style={'padding': '25px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'border': 'none'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.H3(
                                        "📅",
                                        style={
                                            'fontSize': '2.5rem',
                                            'marginBottom': '10px'
                                        }
                                    ),
                                    html.H2(
                                        f"{df['lead_time'].mean():.0f}",
                                        style={
                                            'color': COLORS['primary'],
                                            'fontWeight': 'bold',
                                            'fontSize': '2.5rem',
                                            'marginBottom': '5px'
                                        }
                                    ),
                                    html.P(
                                        "Antecedência Média",
                                        style={
                                            'color': COLORS['dark'],
                                            'marginBottom': '5px',
                                            'fontSize': '16px'
                                        }
                                    ),
                                    html.Small(
                                        "Dias antes do check-in",
                                        style={
                                            'color': COLORS['dark'],
                                            'opacity': '0.7'
                                        }
                                    )
                                ], style={'textAlign': 'center'})
                            ], style={'padding': '25px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                            'border': 'none'
                        })
                    ], width=3)
                ], className="mb-4"),

                # Performance por tipo de hotel
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "🏨 Performance por Tipo de Hotel",
                                style={
                                    'backgroundColor': COLORS['primary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold',
                                    'fontSize': '16px'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Compare o desempenho entre diferentes tipos "
                                    "de estabelecimento:",
                                    style={
                                        'marginBottom': '20px',
                                        'color': COLORS['dark'],
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    figure=make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=(
                                            'Volume de Reservas',
                                            'Taxa de Cancelamento por Hotel'
                                        ),
                                        specs=[[{"type": "bar"}, {"type": "bar"}]]
                                    ).add_trace(
                                        go.Bar(
                                            x=df['hotel'].value_counts().index,
                                            y=df['hotel'].value_counts().values,
                                            marker_color=[
                                                COLORS['primary'],
                                                COLORS['secondary']
                                            ],
                                            text=df['hotel'].value_counts().values,
                                            textposition='auto',
                                            texttemplate='%{text:,}',
                                            hovertemplate=(
                                                '<b>%{x}</b><br>'
                                                'Reservas: %{y:,}<extra></extra>'
                                            ),
                                            showlegend=False
                                        ), row=1, col=1
                                    ).add_trace(
                                        go.Bar(
                                            x=df.groupby('hotel')[
                                                'is_canceled'
                                            ].mean().index,
                                            y=(df.groupby('hotel')[
                                                'is_canceled'
                                            ].mean().values * 100),
                                            marker_color=[
                                                COLORS['accent'],
                                                COLORS['primary']
                                            ],
                                            text=[
                                                f"{v:.1f}%"
                                                for v in (
                                                    df.groupby('hotel')[
                                                        'is_canceled'
                                                    ].mean().values * 100
                                                )
                                            ],
                                            textposition='auto',
                                            hovertemplate=(
                                                '<b>%{x}</b><br>'
                                                'Cancelamento: '
                                                '%{y:.1f}%<extra></extra>'
                                            ),
                                            showlegend=False
                                        ), row=1, col=2
                                    ).update_xaxes(
                                        title_text="", row=1, col=1
                                    ).update_xaxes(
                                        title_text="", row=1, col=2
                                    ).update_yaxes(
                                        title_text="Nº de Reservas", row=1, col=1
                                    ).update_yaxes(
                                        title_text="Taxa de Cancelamento (%)",
                                        row=1, col=2
                                    ).update_layout(
                                        height=400,
                                        showlegend=False,
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        margin=dict(l=20, r=20, t=60, b=20)
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.P(
                                        "💡 Dica: Use essas informações para "
                                        "ajustar estratégias específicas para "
                                        "cada tipo de hotel.",
                                        style={
                                            'color': COLORS['dark'],
                                            'fontSize': '13px',
                                            'fontStyle': 'italic',
                                            'marginBottom': '0'
                                        }
                                    )
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '8px'
                                })
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=12)
                ], className="mb-4"),

                # Análise temporal e receita
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "📅 Sazonalidade das Reservas",
                                style={
                                    'backgroundColor': COLORS['secondary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold',
                                    'fontSize': '16px'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Identifique períodos de alta e baixa demanda "
                                    "ao longo do ano:",
                                    style={
                                        'marginBottom': '20px',
                                        'color': COLORS['dark'],
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    figure=px.histogram(
                                        df, x='arrival_date_month',
                                        color='hotel', barmode='group',
                                        color_discrete_sequence=[
                                            COLORS['primary'],
                                            COLORS['accent']
                                        ],
                                        labels={
                                            'arrival_date_month': 'Mês',
                                            'count': 'Reservas'
                                        }
                                    ).update_layout(
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        showlegend=True,
                                        legend_title_text='Tipo de Hotel',
                                        xaxis_title="Mês de Chegada",
                                        yaxis_title="Número de Reservas",
                                        height=350,
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.P(
                                        "📌 Ação: Planeje promoções e ajustes de "
                                        "preço baseados nos períodos de menor "
                                        "demanda.",
                                        style={
                                            'color': COLORS['dark'],
                                            'fontSize': '13px',
                                            'marginBottom': '0'
                                        }
                                    )
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '8px'
                                })
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "💰 Análise de Receita (ADR)",
                                style={
                                    'backgroundColor': COLORS['primary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold',
                                    'fontSize': '16px'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Distribuição dos valores de diária por tipo "
                                    "de hotel:",
                                    style={
                                        'marginBottom': '20px',
                                        'color': COLORS['dark'],
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    figure=px.box(
                                        df[df['adr'] < 1000], x='hotel', y='adr',
                                        color='hotel',
                                        color_discrete_sequence=[
                                            COLORS['primary'],
                                            COLORS['secondary']
                                        ],
                                        labels={
                                            'hotel': 'Tipo de Hotel',
                                            'adr': 'Valor da Diária (R$)'
                                        }
                                    ).update_layout(
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        showlegend=False,
                                        yaxis_title="Valor da Diária (R$)",
                                        xaxis_title="",
                                        height=350,
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.P(
                                        "💡 Insights: A linha no meio da caixa "
                                        "representa a mediana de preços.",
                                        style={
                                            'color': COLORS['dark'],
                                            'fontSize': '13px',
                                            'marginBottom': '0'
                                        }
                                    )
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '8px'
                                })
                            ], style={'backgroundColor': COLORS['white']})
                        ],
                            style={
                                'borderRadius': '12px',
                                'boxShadow': (
                                    '0 2px 4px rgba(0,0,0,0.1)'
                                )
                        }
                        )
                    ], width=6)
                ], className="mb-4"),

                # Análise geográfica e comportamento
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "🌎 Principais Mercados Geográficos",
                                style={
                                    'backgroundColor': COLORS['secondary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold',
                                    'fontSize': '16px'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Conheça de onde vêm seus hóspedes e "
                                    "identifique oportunidades de mercado:",
                                    style={
                                        'marginBottom': '20px',
                                        'color': COLORS['dark'],
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    figure=px.bar(
                                        eda_results['country_stats'].head(
                                            10
                                        ).reset_index(),
                                        x='count', y='country', orientation='h',
                                        color='count',
                                        color_continuous_scale=[
                                            [0, COLORS['secondary']],
                                            [1, COLORS['primary']]
                                        ],
                                        labels={
                                            'country': 'País',
                                            'count': 'Número de Reservas'
                                        }
                                    ).update_layout(
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        showlegend=False,
                                        yaxis={'categoryorder': 'total ascending'},
                                        xaxis_title="Número de Reservas",
                                        yaxis_title="",
                                        height=350,
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    ).update_traces(
                                        hovertemplate=(
                                            '<b>%{y}</b><br>'
                                            'Reservas: %{x:,}<extra></extra>'
                                        )
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.P(
                                        "🎯 Estratégia: Considere campanhas "
                                        "direcionadas para os mercados mais "
                                        "importantes.",
                                        style={
                                            'color': COLORS['dark'],
                                            'fontSize': '13px',
                                            'marginBottom': '0'
                                        }
                                    )
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '8px'
                                })
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "⏰ Antecedência e Cancelamento",
                                style={
                                    'backgroundColor': COLORS['primary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold',
                                    'fontSize': '16px'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Relação entre tempo de antecedência da "
                                    "reserva e cancelamento:",
                                    style={
                                        'marginBottom': '20px',
                                        'color': COLORS['dark'],
                                        'fontSize': '14px'
                                    }
                                ),
                                dcc.Graph(
                                    figure=px.box(
                                        df, x='is_canceled', y='lead_time',
                                        color='is_canceled',
                                        color_discrete_sequence=[
                                            COLORS['secondary'],
                                            COLORS['accent']
                                        ],
                                        labels={
                                            'is_canceled': 'Status',
                                            'lead_time': 'Antecedência (dias)'
                                        }
                                    ).update_layout(
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        showlegend=False,
                                        yaxis_title="Antecedência (dias)",
                                        xaxis_title="",
                                        height=350,
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    ).update_xaxes(
                                        ticktext=[
                                            'Reservas Mantidas',
                                            'Reservas Canceladas'
                                        ],
                                        tickvals=[0, 1]
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.P(
                                        "📊 Observe: Reservas com muita "
                                        "antecedência tendem a ter maior taxa de "
                                        "cancelamento.",
                                        style={
                                            'color': COLORS['dark'],
                                            'fontSize': '13px',
                                            'marginBottom': '0'
                                        }
                                    )
                                ], style={
                                    'padding': '10px',
                                    'backgroundColor': COLORS['background'],
                                    'borderRadius': '8px'
                                })
                            ], style={'backgroundColor': COLORS['white']})
                        ],
                            style={
                                'borderRadius': '12px',
                                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        }
                        )
                    ], width=6)
                ])
            ]),

        # Tab 2: Previsão de Cancelamentos (Business-focused)
        dcc.Tab(
            label='🎯 Previsão de Cancelamentos',
            style={'padding': '10px', 'fontWeight': 'bold'},
            children=[

                # Header explicativo
                dbc.Row([
                    dbc.Col([
                        dbc.Alert([
                            html.H4(
                                "💡 Sistema Inteligente de Previsão de "
                                "Cancelamentos",
                                className="alert-heading"
                            ),
                            html.P(
                                "Antecipe cancelamentos e tome decisões "
                                "estratégicas para maximizar sua receita e "
                                "ocupação."
                            ),
                            html.Hr(),
                            html.P(
                                "Nosso sistema analisa padrões históricos "
                                "para identificar reservas com maior risco "
                                "de cancelamento, permitindo que você aja "
                                "proativamente.",
                                className="mb-0"
                            )
                        ], color="info", style={'borderRadius': '12px'})
                    ], width=12)
                ], className="mb-4"),

                # Cards de métricas principais
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(
                                    "📊",
                                    style={
                                        'fontSize': '2rem',
                                        'marginBottom': '10px'
                                    }
                                ),
                                html.H4(
                                    (
                                        f"{ml_results_parts:.1f}%"
                                    ),
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '2rem'
                                    }
                                ),
                                html.P(
                                    "Taxa de Acerto",
                                    style={
                                        'color': COLORS['dark'],
                                        'marginBottom': '5px'
                                    }
                                ),
                                html.Small(
                                    "Previsões corretas sobre cancelamentos",
                                    style={
                                        'color': COLORS['dark'],
                                        'opacity': '0.7'
                                    }
                                )
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(
                                    "🎯",
                                    style={
                                        'fontSize': '2rem',
                                        'marginBottom': '10px'
                                    }
                                ),
                                html.H4(
                                    f"{df['is_canceled'].mean() * 100:.1f}%",
                                    style={
                                        'color': COLORS['accent'],
                                        'fontWeight': 'bold',
                                        'fontSize': '2rem'
                                    }
                                ),
                                html.P(
                                    "Taxa de Cancelamento",
                                    style={
                                        'color': COLORS['dark'],
                                        'marginBottom': '5px'
                                    }
                                ),
                                html.Small(
                                    "Média histórica do seu hotel",
                                    style={
                                        'color': COLORS['dark'],
                                        'opacity': '0.7'
                                    }
                                )
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(
                                    "💰",
                                    style={
                                        'fontSize': '2rem',
                                        'marginBottom': '10px'
                                    }
                                ),
                                html.H4(
                                    f"${df['adr'].mean():.0f}",
                                    style={
                                        'color': COLORS['secondary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '2rem'
                                    }
                                ),
                                html.P(
                                    "Diária Média",
                                    style={
                                        'color': COLORS['dark'],
                                        'marginBottom': '5px'
                                    }
                                ),
                                html.Small(
                                    "Valor médio por reserva",
                                    style={
                                        'color': COLORS['dark'],
                                        'opacity': '0.7'
                                    }
                                )
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=3),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H3(
                                    "⚡",
                                    style={
                                        'fontSize': '2rem',
                                        'marginBottom': '10px'
                                    }
                                ),
                                html.H4(
                                    f"{df['lead_time'].mean():.0f} dias",
                                    style={
                                        'color': COLORS['primary'],
                                        'fontWeight': 'bold',
                                        'fontSize': '2rem'
                                    }
                                ),
                                html.P(
                                    "Antecedência Média",
                                    style={
                                        'color': COLORS['dark'],
                                        'marginBottom': '5px'
                                    }
                                ),
                                html.Small(
                                    "Tempo médio de reserva",
                                    style={
                                        'color': COLORS['dark'],
                                        'opacity': '0.7'
                                    }
                                )
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ], style={
                            'borderRadius': '12px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                        })
                    ], width=3)
                ], className="mb-4"),

                # Principais fatores de cancelamento
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                "🔍 Principais Fatores que Influenciam "
                                "Cancelamentos",
                                style={
                                    'backgroundColor': COLORS['secondary'],
                                    'color': COLORS['white'],
                                    'fontWeight': 'bold'
                                }
                            ),
                            dbc.CardBody([
                                html.P(
                                    "Identificamos os fatores mais importantes que determinam se uma reserva será cancelada:",
                                    style={'marginBottom': '20px', 'color': COLORS['dark']}
                                ),
                                dcc.Graph(
                                    figure=px.bar(
                                        feature_importance_df.head(10) if feature_importance_df is not None else pd.DataFrame(
                                            {'feature': ['N/A'], 'importance': [0]}),
                                        x='importance', y='feature',
                                        orientation='h',
                                        color='importance',
                                        color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        yaxis={'categoryorder': 'total ascending'},
                                        showlegend=False,
                                        xaxis_title="Relevância",
                                        yaxis_title="",
                                        margin=dict(l=20, r=20, t=20, b=20)
                                    ).update_traces(
                                        hovertemplate='<b>%{y}</b><br>Importância: %{x:.2%}<extra></extra>'
                                    )
                                )
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={'borderRadius': '12px'})
                    ], width=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("💼 Ações Recomendadas",
                                           style={
                                               'backgroundColor': COLORS['primary'],
                                               'color': COLORS['white'],
                                               'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.Div([
                                    html.Div([
                                        html.H5("🎯 Para Reservas de Alto Risco:", style={'color': COLORS['accent'], 'marginBottom': '15px'}),
                                        html.Ul([
                                            html.Li("Envie lembretes personalizados 7 dias antes do check-in", style={'marginBottom': '8px'}),
                                            html.Li("Ofereça upgrades ou benefícios para incentivar confirmação", style={'marginBottom': '8px'}),
                                            html.Li("Entre em contato direto para confirmar a reserva", style={'marginBottom': '8px'}),
                                            html.Li("Considere política de cancelamento mais flexível", style={'marginBottom': '8px'})
                                        ], style={'color': COLORS['dark']})
                                    ], style={'marginBottom': '25px'}),

                                    html.Div([
                                        html.H5("💰 Gestão de Receita:", style={'color': COLORS['secondary'], 'marginBottom': '15px'}),
                                        html.Ul([
                                            html.Li("Ajuste preços baseado no perfil de risco", style={'marginBottom': '8px'}),
                                            html.Li("Mantenha lista de espera para períodos de alta demanda", style={'marginBottom': '8px'}),
                                            html.Li("Implemente tarifas não-reembolsáveis com desconto", style={'marginBottom': '8px'}),
                                            html.Li("Monitore padrões sazonais de cancelamento", style={'marginBottom': '8px'})
                                        ], style={'color': COLORS['dark']})
                                    ])
                                ], style={'padding': '10px'})
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={'borderRadius': '12px'})
                    ], width=6)
                ], className="mb-4"),

                # Perfis de clientes
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("👥 Perfis de Clientes Identificados",
                                           style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                            dbc.CardBody([
                                html.P(
                                    "Nosso sistema identificou diferentes perfis de clientes com comportamentos distintos:",
                                    style={'marginBottom': '20px', 'color': COLORS['dark']}
                                ),
                                dcc.Graph(
                                    figure=make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Características por Perfil', 'Distribuição de Clientes'),
                                        specs=[[{"type": "table"}, {"type": "scatter"}]],
                                        column_widths=[0.5, 0.5]
                                    ).add_trace(
                                        go.Table(
                                            header=dict(
                                                values=['Perfil', 'Risco de Cancelamento', 'Gasto Médio', 'Antecedência'],
                                                fill_color=COLORS['secondary'],
                                                font=dict(color=COLORS['white'], size=11, family='Arial'),
                                                align='center',
                                                height=30
                                            ),
                                            cells=dict(
                                                values=[
                                                    clustering_results['cluster_labels'],
                                                    [f"{v:.0%}" for v in clustering_results['cluster_analysis']['is_canceled']],
                                                    [f"${v:.0f}" for v in clustering_results['cluster_analysis']['adr']],
                                                    [f"{v:.0f} dias" for v in clustering_results['cluster_analysis']['lead_time']]
                                                ],
                                                fill_color=COLORS['background'],
                                                font=dict(color=COLORS['dark'], size=11),
                                                align='center',
                                                height=28
                                            )
                                        ), row=1, col=1
                                    ).add_trace(
                                        go.Scatter(
                                            x=clustering_results['X_pca'][clustering_results['clusters'] == 0, 0],
                                            y=clustering_results['X_pca'][clustering_results['clusters'] == 0, 1],
                                            mode='markers',
                                            name=clustering_results['cluster_labels'][0],
                                            marker=dict(
                                                color='#1f77b4',  # Azul
                                                size=8,
                                                opacity=0.6,
                                                line=dict(width=0.5, color='white')
                                            ),
                                            hovertemplate='<b>' + clustering_results['cluster_labels'][0] + '</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=2
                                    ).add_trace(
                                        go.Scatter(
                                            x=clustering_results['X_pca'][clustering_results['clusters'] == 1, 0],
                                            y=clustering_results['X_pca'][clustering_results['clusters'] == 1, 1],
                                            mode='markers',
                                            name=clustering_results['cluster_labels'][1],
                                            marker=dict(
                                                color='#2ca02c',  # Verde
                                                size=8,
                                                opacity=0.6,
                                                line=dict(width=0.5, color='white')
                                            ),
                                            hovertemplate='<b>' + clustering_results['cluster_labels'][1] + '</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=2
                                    ).add_trace(
                                        go.Scatter(
                                            x=clustering_results['X_pca'][clustering_results['clusters'] == 2, 0],
                                            y=clustering_results['X_pca'][clustering_results['clusters'] == 2, 1],
                                            mode='markers',
                                            name=clustering_results['cluster_labels'][2],
                                            marker=dict(
                                                color='#ff7f0e',  # Laranja
                                                size=8,
                                                opacity=0.6,
                                                line=dict(width=0.5, color='white')
                                            ),
                                            hovertemplate='<b>' + clustering_results['cluster_labels'][2] + '</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=2
                                    ).update_xaxes(
                                        title_text=f"PC1 ({clustering_results['variance_explained'][0]:.1%} var.)",
                                        row=1, col=2,
                                        showgrid=True,
                                        gridcolor='#E5E5E5',
                                        gridwidth=1,
                                        zeroline=True,
                                        zerolinecolor='#CCCCCC',
                                        zerolinewidth=2
                                    ).update_yaxes(
                                        title_text=f"PC2 ({clustering_results['variance_explained'][1]:.1%} var.)",
                                        row=1, col=2,
                                        showgrid=True,
                                        gridcolor='#E5E5E5',
                                        gridwidth=1,
                                        zeroline=True,
                                        zerolinecolor='#CCCCCC',
                                        zerolinewidth=2
                                    ).update_layout(
                                        height=450,
                                        showlegend=True,
                                        legend=dict(
                                            title=dict(text="Perfil de Cliente", font=dict(size=12, color=COLORS['dark'])),
                                            orientation="v",
                                            yanchor="top",
                                            y=0.98,
                                            xanchor="right",
                                            x=1.18,
                                            bgcolor="rgba(255,255,255,0.9)",
                                            bordercolor=COLORS['dark'],
                                            borderwidth=1
                                        ),
                                        plot_bgcolor='white',
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark'],
                                        margin=dict(l=20, r=120, t=40, b=20)
                                    )
                                ),
                                html.Hr(),
                                html.Div([
                                    html.H6("📌 Como usar essa informação:", style={'color': COLORS['secondary'], 'marginBottom': '10px'}),
                                    html.P(
                                        "• Personalize comunicação e ofertas para cada perfil de cliente",
                                        style={'color': COLORS['dark'], 'marginBottom': '5px'}
                                    ),
                                    html.P(
                                        "• Ajuste estratégias de retenção baseado no perfil da reserva",
                                        style={'color': COLORS['dark'], 'marginBottom': '5px'}
                                    ),
                                    html.P(
                                        "• Identifique oportunidades de upselling em perfis de maior valor",
                                        style={'color': COLORS['dark']}
                                    )
                                ], style={'padding': '15px', 'backgroundColor': COLORS['background'], 'borderRadius': '8px'})
                            ], style={'backgroundColor': COLORS['white']})
                        ], style={'borderRadius': '12px'})
                    ], width=12)
                ])
            ]),

        # Tab 3: Simulação de Cancelamento
        dcc.Tab(label='🎲 Simulador de Cancelamento', style={'padding': '10px', 'fontWeight': 'bold'}, children=[

            # Banner explicativo
            dbc.Row([
                dbc.Col([
                    dbc.Alert([
                        html.H5("🎲 Simulador de Risco de Cancelamento", className="alert-heading"),
                        html.P("Simule diferentes cenários de reserva e descubra a probabilidade de cancelamento. "
                               "Use esta ferramenta para avaliar o risco antes de confirmar uma reserva."),
                    ], color="info", style={'borderRadius': '12px'})
                ], width=12)
            ], className="mb-4"),

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
        ])
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
            html.P("👆 Preencha os dados da reserva acima",
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
# 7. EXECUÇÃO DO DASHBOARD - VERSÃO CORRIGIDA
# ============================================================================

print("\n" + "=" * 80)
print("🚀 INICIANDO SERVIDOR DASHBOARD")
print("=" * 80)

if __name__ == '__main__':
    print("📋 Dashboard pronto para execução!")
    print("⚠️  No Google Colab, use o seguinte comando para visualizar:")
    print("    from google.colab.output import eval_js")
    print("    print(eval_js(\"google.colab.kernel.proxyPort(8050)\"))")
    print("\n🎯 Executando servidor...")

    # CORREÇÃO: Usar app.run() em vez de app.run_server()
    app.run(debug=False, host='0.0.0.0', port=8050)
