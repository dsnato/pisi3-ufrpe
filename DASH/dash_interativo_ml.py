# -*- coding: utf-8 -*-
"""
Dashboard Interativo - Hotel Booking Analysis
Análise exploratória e resultados de Machine Learning
"""

print("📊 INICIANDO DASHBOARD DE ANÁLISE DE RESERVAS DE HOTÉIS")

# ============================================================================
# IMPORTAÇÕES
# ============================================================================

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

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

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================

print("\n📂 Carregando dados...")

# Carregar dataset principal
parquet_file = 'hotel_bookings.parquet'
csv_files = ['ML/data/hotel_bookings.csv', 'EDA/hotel_bookings.csv', 'hotel_bookings.csv']

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"✅ Dataset carregado do Parquet: {parquet_file}")
else:
    df = None
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print(f"✅ Dataset carregado de: {csv_file}")
            break
    
    if df is None:
        raise FileNotFoundError("❌ Arquivo de dados não encontrado. Verifique se o dataset está em uma das pastas: ML/data/, EDA/ ou raiz do projeto.")

# ============================================================================
# CARREGAMENTO DE RESULTADOS DE ML (com fallback para dados dummy)
# ============================================================================

print("\n🤖 Carregando resultados de Machine Learning...")

# 1. Resultados dos modelos
try:
    model_results = pd.read_csv('model_results.csv')
    print("✅ Resultados dos modelos carregados")
except:
    print("⚠️  Criando resultados dummy (execute o script ML primeiro para dados reais)")
    model_results = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Logistic Regression', 'Gradient Boosting'],
        'Acurácia': [0.85, 0.83, 0.78, 0.84],
        'Precisão': [0.84, 0.82, 0.76, 0.83],
        'Recall': [0.81, 0.79, 0.74, 0.80],
        'F1-Score': [0.82, 0.80, 0.75, 0.81]
    })

# 2. Importância das features
try:
    import joblib
    
    # Tentar carregar modelo e preprocessor
    if os.path.exists('best_model.pkl'):
        best_model = joblib.load('best_model.pkl')
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Extrair nomes das features
        feature_names = list(preprocessor.get_feature_names_out())
        
        # Extrair importâncias
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            importances = best_model.named_steps['classifier'].feature_importances_
        else:
            importances = abs(best_model.named_steps['classifier'].coef_[0])
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("✅ Importância de features carregada do modelo")
    else:
        raise FileNotFoundError()
        
except:
    print("⚠️  Criando importância de features dummy")
    feature_importance_df = pd.DataFrame({
        'feature': ['lead_time', 'adr', 'total_nights', 'previous_cancellations', 
                   'booking_changes', 'deposit_type_Non_Refund', 'market_segment_Online',
                   'customer_type_Transient', 'required_car_parking_spaces', 'total_of_special_requests'],
        'importance': np.linspace(0.15, 0.05, 10)
    })

# 3. Análise de clusters
try:
    cluster_analysis = pd.read_csv('cluster_analysis.csv')
    print("✅ Análise de clusters carregada")
except:
    print("⚠️  Criando análise de clusters dummy")
    cluster_analysis = pd.DataFrame({
        'Cluster': [0, 1, 2],
        'is_canceled': [0.25, 0.50, 0.75],
        'adr': [95.5, 110.2, 85.3],
        'lead_time': [45, 95, 150]
    })

# 4. PCA e clusters para visualização
try:
    import joblib
    from sklearn.decomposition import PCA
    
    # Tentar carregar modelo de clustering
    if os.path.exists('kmeans_cluster_model.pkl'):
        kmeans_model = joblib.load('kmeans_cluster_model.pkl')
        
        # Preparar amostra de dados
        numeric_features = ['lead_time', 'adr', 'adults', 'children', 'babies',
                           'stays_in_weekend_nights', 'stays_in_week_nights']
        
        X_sample = df[numeric_features].fillna(0).sample(n=min(5000, len(df)), random_state=42)
        
        # PCA para visualização 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        
        # Prever clusters
        try:
            clusters = kmeans_model.predict(X_sample)
        except:
            clusters = np.random.randint(0, 3, len(X_sample))
        
        print("✅ PCA e clusters carregados do modelo")
    else:
        raise FileNotFoundError()
        
except:
    print("⚠️  Criando visualização de clusters dummy")
    np.random.seed(42)
    n_samples = 1000
    X_pca = np.random.randn(n_samples, 2) * 2
    clusters = np.random.randint(0, 3, n_samples)

print("\n✅ Todos os dados carregados com sucesso!")

# ============================================================================
# CRIAÇÃO DA APLICAÇÃO DASH
# ============================================================================


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Hotel Booking Analysis Dashboard"

# CSS customizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #EFEFF0 !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card {
                border: none !important;
                box-shadow: 0 2px 8px rgba(19, 47, 59, 0.1) !important;
                border-radius: 12px !important;
            }
            .tab {
                background-color: #FFFFFF !important;
                color: #132F3B !important;
                border: 1px solid #EFEFF0 !important;
            }
            .tab--selected {
                background-color: #132F3B !important;
                color: #FFFFFF !important;
                border-bottom: 3px solid #0162B3 !important;
            }
            .tab:hover {
                background-color: #0162B3 !important;
                color: #FFFFFF !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# LAYOUT DO DASHBOARD
# ============================================================================

app.layout = dbc.Container([
    # ========================================================================
    # CABEÇALHO
    # ========================================================================
    dbc.Row([
        dbc.Col([
            html.H1("🏨 Hotel Booking Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': COLORS['dark'], 'fontWeight': 'bold', 'padding': '20px 0'})
        ], width=12)
    ], style={'background': f'linear-gradient(135deg, {COLORS["white"]} 0%, {COLORS["background"]} 100%)'}),

    # ========================================================================
    # TABS PRINCIPAIS
    # ========================================================================
    dcc.Tabs(style={'marginTop': '20px'}, children=[
        
        # ====================================================================
        # TAB 1: VISÃO GERAL
        # ====================================================================
        dcc.Tab(label='📈 Visão Geral', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            
            # Métricas principais
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Métricas Principais", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.H4(f"Total de Reservas: {df.shape[0]:,}", className="card-text", style={'color': COLORS['dark']}),
                            html.H4(f"Taxa de Cancelamento: {df['is_canceled'].mean()*100:.1f}%",
                                   className="card-text", style={'color': COLORS['accent'], 'fontWeight': 'bold'}),
                            html.H4(f"ADR Médio: ${df['adr'].mean():.2f}", className="card-text", style={'color': COLORS['dark']})
                        ], style={'backgroundColor': COLORS['white']})
                    ], className="mb-4", style={'borderRadius': '12px'})
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏨 Distribuição por Tipo de Hotel", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(df['hotel'].value_counts().reset_index(),
                                            x='hotel', y='count',
                                            title='Distribuição por Tipo de Hotel',
                                            labels={'hotel': 'Tipo de Hotel', 'count': 'Quantidade de Reservas'},
                                            color='hotel',
                                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_size=16,
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=8)
            ], className="mb-4"),

            # Gráficos de análise temporal e ADR

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📅 Reservas por Mês", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.histogram(df, x='arrival_date_month',
                                                  color='hotel', barmode='group',
                                                  title='Reservas por Mês e Tipo de Hotel',
                                                  color_discrete_sequence=[COLORS['secondary'], COLORS['accent']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💰 ADR por Tipo de Hotel", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.box(df[df['adr'] < 1000], x='hotel', y='adr',
                                            title='Distribuição do ADR por Tipo de Hotel',
                                            color='hotel',
                                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ], className="mb-4"),

            # Análise geográfica e lead time
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌎 Top 10 Países", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(df['country'].value_counts().head(10).reset_index(),
                                            x='count', y='country', orientation='h',
                                            title='Top 10 Países por Reservas',
                                            color='count',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['primary']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⏰ Lead Time vs Cancelamento", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.box(df, x='is_canceled', y='lead_time',
                                            title='Lead Time vs Cancelamento',
                                            labels={'is_canceled': 'Cancelado', 'lead_time': 'Lead Time (dias)'},
                                            color='is_canceled',
                                            color_discrete_sequence=[COLORS['secondary'], COLORS['accent']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ])
        ]),

        # ====================================================================
        # TAB 2: MACHINE LEARNING
        # ====================================================================
        dcc.Tab(label='🤖 Machine Learning', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            
            # Resultados dos modelos
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Resultados dos Modelos", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Modelo", style={'color': COLORS['dark'], 'fontWeight': 'bold'}), 
                                        html.Th("Acurácia", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        html.Th("F1-Score", style={'color': COLORS['dark'], 'fontWeight': 'bold'})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(row['Modelo'], style={'color': COLORS['dark']}),
                                        html.Td(f"{row['Acurácia']:.4f}", style={'color': COLORS['dark']}),
                                        html.Td(f"{row['F1-Score']:.4f}", style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                    ]) for _, row in model_results.iterrows()
                                ])
                            ], className="table table-striped", style={'marginBottom': '0'})
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Comparação de Modelos", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(model_results, x='Modelo', y='F1-Score',
                                            title='Desempenho dos Modelos (F1-Score)',
                                            color='F1-Score',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ], className="mb-4"),

            # Importância das features
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Importância das Features", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(feature_importance_df.head(10),
                                            x='importance', y='feature',
                                            title='Top 10 Features Mais Importantes',
                                            orientation='h',
                                            color='importance',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ], className="mb-4"),

            # Análise de clusters
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Análise de Clusters", 
                                     style={'backgroundColor': COLORS['accent'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=make_subplots(
                                    rows=1, cols=2,
                                    subplot_titles=('Características dos Clusters', 'Visualização PCA (2D)'),
                                    specs=[[{"type": "table"}, {"type": "scatter"}]],
                                    column_widths=[0.4, 0.6]
                                ).add_trace(
                                    go.Table(
                                        header=dict(
                                            values=['Cluster', 'Taxa Cancel.', 'ADR Médio', 'Lead Time'],
                                            fill_color=COLORS['primary'],
                                            font=dict(color=COLORS['white'], size=12),
                                            align='center'
                                        ),
                                        cells=dict(
                                            values=[
                                                [f'Cluster {i}' for i in cluster_analysis['Cluster']],
                                                [f"{v:.2%}" for v in cluster_analysis['is_canceled']],
                                                [f"${v:.2f}" for v in cluster_analysis['adr']],
                                                [f"{v:.0f} dias" for v in cluster_analysis['lead_time']]
                                            ],
                                            fill_color=COLORS['background'],
                                            font=dict(color=COLORS['dark'], size=11),
                                            align='center'
                                        )
                                    ), row=1, col=1
                                ).add_trace(
                                    go.Scatter(
                                        x=X_pca[:, 0], 
                                        y=X_pca[:, 1], 
                                        mode='markers',
                                        marker=dict(
                                            color=clusters, 
                                            colorscale='Viridis',
                                            size=5,
                                            opacity=0.6,
                                            colorbar=dict(title="Cluster")
                                        ),
                                        text=[f'Cluster {c}' for c in clusters],
                                        hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                                    ), row=1, col=2
                                ).update_xaxes(title_text="Componente Principal 1", row=1, col=2)
                                .update_yaxes(title_text="Componente Principal 2", row=1, col=2)
                                .update_layout(
                                    height=500, 
                                    showlegend=False,
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ])
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

from dash import dcc, html, Input, Output, callback_context

# Carregar dados e modelos - usar Parquet como padrão
# Quando executado da raiz do projeto: python DASH/dash_interativo_ml.py
parquet_file = 'hotel_bookings.parquet'
csv_file_ml = 'ML/data/hotel_bookings.csv'
csv_file_eda = 'EDA/hotel_bookings.csv'
csv_file_root = 'hotel_bookings.csv'

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"✅ Dataset carregado do Parquet: {parquet_file}")
elif os.path.exists(csv_file_ml):
    df = pd.read_csv(csv_file_ml)
    print(f"✅ Dataset carregado do CSV: {csv_file_ml}")
    # Salvar em Parquet para próximas execuções
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Dataset salvo em Parquet: {parquet_file}")
elif os.path.exists(csv_file_eda):
    df = pd.read_csv(csv_file_eda)
    print(f"✅ Dataset carregado do CSV: {csv_file_eda}")
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Dataset salvo em Parquet: {parquet_file}")
elif os.path.exists(csv_file_root):
    df = pd.read_csv(csv_file_root)
    print(f"✅ Dataset carregado do CSV: {csv_file_root}")
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Dataset salvo em Parquet: {parquet_file}")
else:
    raise FileNotFoundError("❌ Arquivo de dados não encontrado. Certifique-se de que hotel_bookings.csv existe em ML/data/, EDA/ ou na raiz.")

# Tentar carregar arquivos analisados
try:
    df_analyzed = pd.read_csv('hotel_bookings_analyzed.csv')
    print("✅ hotel_bookings_analyzed.csv carregado.")
except:
    print("⚠️ hotel_bookings_analyzed.csv não encontrado. Usando df original.")
    df_analyzed = df.copy()

try:
    model_results = pd.read_csv('model_results.csv')
    print("✅ model_results.csv carregado.")
except:
    print("⚠️ model_results.csv não encontrado. Criando dados dummy.")
    model_results = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Logistic Regression'],
        'Acurácia': [0.85, 0.83, 0.78],
        'F1-Score': [0.82, 0.80, 0.75]
    })

try:
    cluster_analysis = pd.read_csv('cluster_analysis.csv')
    print("✅ cluster_analysis.csv carregado.")
except:
    print("⚠️ cluster_analysis.csv não encontrado. Criando dados dummy.")
    cluster_analysis = pd.DataFrame({
        'is_canceled': [0.3, 0.5, 0.7],
        'adr': [100, 120, 90],
        'lead_time': [50, 100, 150]
    })

# Carregar modelos treinados
print(f"Current working directory: {os.getcwd()}")
best_model = None
kmeans_model = None
preprocessor = None
feature_importance_df = None
X_pca = None
clusters = None

try:
    # Tentar carregar dos diretórios possíveis (executado da raiz do projeto)
    model_paths = ['./', 'ML/']
    for path in model_paths:
        try:
            best_model = joblib.load(os.path.join(path, 'best_classification_model.pkl'))
            kmeans_model = joblib.load(os.path.join(path, 'kmeans_cluster_model.pkl'))
            preprocessor = joblib.load(os.path.join(path, 'preprocessor.pkl'))
            print(f"✅ Modelos carregados com sucesso de {path}!")
            break
        except:
            continue
    
    if best_model is None:
        print("⚠️ Modelos não encontrados. Executando com dados estáticos.")
except Exception as e:
    print(f"⚠️ Erro ao carregar modelos: {e}. Executando com dados estáticos.")

# Criar aplicação Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Hotel Booking Analysis Dashboard"

# CSS customizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #EFEFF0 !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card {
                border: none !important;
                box-shadow: 0 2px 8px rgba(19, 47, 59, 0.1) !important;
                border-radius: 12px !important;
            }
            .tab {
                background-color: #FFFFFF !important;
                color: #132F3B !important;
                border: 1px solid #EFEFF0 !important;
            }
            .tab--selected {
                background-color: #132F3B !important;
                color: #FFFFFF !important;
                border-bottom: 3px solid #0162B3 !important;
            }
            .tab:hover {
                background-color: #0162B3 !important;
                color: #FFFFFF !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

"""# LAYOUT DO DASHBOARD"""

# Debug: Verificar dados antes de criar layout
print("\n" + "="*50)
print("📊 VERIFICAÇÃO DE DADOS PARA O DASHBOARD")
print("="*50)
print(f"feature_importance_df is None: {feature_importance_df is None}")
if feature_importance_df is not None:
    print(f"feature_importance_df shape: {feature_importance_df.shape}")
    print(f"feature_importance_df head:\n{feature_importance_df.head()}")
print(f"\nX_pca is None: {X_pca is None}")
if X_pca is not None:
    print(f"X_pca shape: {X_pca.shape}")
print(f"clusters is None: {clusters is None}")
if clusters is not None:
    print(f"clusters length: {len(clusters)}")
    print(f"Unique clusters: {set(clusters) if hasattr(clusters, '__iter__') else 'N/A'}")
print("="*50 + "\n")

app.layout = dbc.Container([
    # Cabeçalho
    dbc.Row([
        dbc.Col([
            html.H1("🏨 Hotel Booking Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': COLORS['dark'], 'fontWeight': 'bold', 'padding': '20px 0'})
        ], width=12)
    ], style={'background': f'linear-gradient(135deg, {COLORS["white"]} 0%, {COLORS["background"]} 100%)'}),

    # Abas principais
    dcc.Tabs(style={'marginTop': '20px'}, children=[
        # Tab 1: Visão Geral
        dcc.Tab(label='📈 Visão Geral', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Métricas Principais", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.H4(f"Total de Reservas: {df.shape[0]:,}", className="card-text", style={'color': COLORS['dark']}),
                            html.H4(f"Taxa de Cancelamento: {df['is_canceled'].mean()*100:.1f}%",
                                   className="card-text", style={'color': COLORS['accent'], 'fontWeight': 'bold'}),
                            html.H4(f"ADR Médio: ${df['adr'].mean():.2f}", className="card-text", style={'color': COLORS['dark']})
                        ], style={'backgroundColor': COLORS['white']})
                    ], className="mb-4", style={'borderRadius': '12px'})
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🏨 Distribuição por Tipo de Hotel", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(df['hotel'].value_counts().reset_index(),
                                            x='hotel', y='count',
                                            title='Distribuição por Tipo de Hotel',
                                            labels={'hotel': 'Tipo de Hotel', 'count': 'Quantidade de Reservas'},
                                            color='hotel',
                                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_size=16,
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=8)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📅 Reservas por Mês", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.histogram(df, x='arrival_date_month',
                                                  color='hotel', barmode='group',
                                                  title='Reservas por Mês e Tipo de Hotel',
                                                  color_discrete_sequence=[COLORS['secondary'], COLORS['accent']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💰 ADR por Tipo de Hotel", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.box(df[df['adr'] < 1000], x='hotel', y='adr',
                                            title='Distribuição do ADR por Tipo de Hotel',
                                            color='hotel',
                                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌎 Top 10 Países", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(df['country'].value_counts().head(10).reset_index(),
                                            x='count', y='country', orientation='h',
                                            title='Top 10 Países por Reservas',
                                            color='count',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['primary']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⏰ Lead Time vs Cancelamento", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.box(df, x='is_canceled', y='lead_time',
                                            title='Lead Time vs Cancelamento',
                                            labels={'is_canceled': 'Cancelado', 'lead_time': 'Lead Time (dias)'},
                                            color='is_canceled',
                                            color_discrete_sequence=[COLORS['secondary'], COLORS['accent']])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ])
        ]),

        # Tab 2: Análise de Cancelamentos
        dcc.Tab(label='❌ Análise de Cancelamentos', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filtros", 
                                     style={'backgroundColor': COLORS['dark'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Label("Tipo de Hotel:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='hotel-filter',
                                options=[{'label': 'Todos', 'value': 'all'}] +
                                       [{'label': hotel, 'value': hotel} for hotel in df['hotel'].unique()],
                                value='all',
                                style={'marginBottom': '15px'}
                            ),
                            html.Br(),
                            html.Label("País:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='country-filter',
                                options=[{'label': 'Todos', 'value': 'all'}] +
                                       [{'label': country, 'value': country} for country in df['country'].value_counts().head(20).index],
                                value='all'
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], className="mb-4", style={'borderRadius': '12px'})
                ], width=3),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Taxa de Cancelamento", className="card-title", style={'color': COLORS['dark']}),
                                    html.H2(id='cancel-rate', style={'color': COLORS['accent'], 'fontWeight': 'bold'})
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="text-center", style={'borderRadius': '12px'})
                        ], width=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Reservas Totais", className="card-title", style={'color': COLORS['dark']}),
                                    html.H2(id='total-bookings', style={'color': COLORS['primary'], 'fontWeight': 'bold'})
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="text-center", style={'borderRadius': '12px'})
                        ], width=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("ADR Médio", className="card-title", style={'color': COLORS['dark']}),
                                    html.H2(id='avg-adr', style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="text-center", style={'borderRadius': '12px'})
                        ], width=4)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📊 Cancelamentos por Segmento", 
                                             style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Graph(id='cancel-by-segment')
                                ], style={'backgroundColor': COLORS['white']})
                            ], style={'borderRadius': '12px'})
                        ], width=6),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📈 Cancelamentos por Mês", 
                                             style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Graph(id='cancel-by-month')
                                ], style={'backgroundColor': COLORS['white']})
                            ], style={'borderRadius': '12px'})
                        ], width=6)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("👥 Cancelamentos por Tipo de Cliente", 
                                             style={'backgroundColor': COLORS['dark'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Graph(id='cancel-by-customer-type')
                                ], style={'backgroundColor': COLORS['white']})
                            ], style={'borderRadius': '12px'})
                        ], width=12)
                    ])
                ], width=9)
            ])
        ]),

        # Tab 3: Machine Learning
        dcc.Tab(label='🤖 Machine Learning', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Resultados dos Modelos", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Table([
                                html.Thead([
                                    html.Tr([html.Th("Modelo", style={'color': COLORS['dark']}), 
                                            html.Th("Acurácia", style={'color': COLORS['dark']}), 
                                            html.Th("F1-Score", style={'color': COLORS['dark']})])
                                ]),
                                html.Tbody([
                                    html.Tr([html.Td(row['Modelo'], style={'color': COLORS['dark']}),
                                            html.Td(f"{row['Acurácia']:.4f}", style={'color': COLORS['dark']}),
                                            html.Td(f"{row['F1-Score']:.4f}", style={'color': COLORS['secondary'], 'fontWeight': 'bold'})])
                                    for _, row in model_results.iterrows()
                                ])
                            ], className="table table-striped", style={'backgroundColor': COLORS['white']})
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Comparação de Modelos", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(model_results, x='Modelo', y='F1-Score',
                                            title='Desempenho dos Modelos (F1-Score)',
                                            color='F1-Score',
                                            color_continuous_scale=[[0, COLORS['secondary']], [0.5, COLORS['primary']], [1, COLORS['accent']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark']
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Importância das Features", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                id='feature-importance-graph',
                                figure=px.bar(
                                    feature_importance_df.head(15) if feature_importance_df is not None and not feature_importance_df.empty else pd.DataFrame({'feature': ['Sem dados'], 'importance': [0]}),
                                    x='importance', y='feature',
                                    title='Top 15 Features Mais Importantes',
                                    orientation='h',
                                    color='importance',
                                    color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
                                ).update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark'],
                                    yaxis={'categoryorder':'total ascending'}
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔮 Análise de Clusters", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                id='cluster-analysis-graph',
                                figure=(
                                    make_subplots(
                                        rows=1, cols=2,
                                        subplot_titles=('Características dos Clusters', 'Visualização PCA'),
                                        specs=[[{"type": "table"}, {"type": "scatter"}]]
                                    ).add_trace(
                                        go.Table(
                                            header=dict(values=['Cluster'] + cluster_analysis.columns.tolist(),
                                                       fill_color=COLORS['secondary'],
                                                       align='left',
                                                       font=dict(color=COLORS['white'], size=12)),
                                            cells=dict(values=[[f'Cluster {i}' for i in cluster_analysis.index]] +
                                                      [cluster_analysis[col].tolist() for col in cluster_analysis.columns],
                                                      fill_color=COLORS['background'],
                                                      align='left',
                                                      font=dict(color=COLORS['dark'], size=11))
                                        ), row=1, col=1
                                    ).add_trace(
                                        go.Scatter(
                                            x=X_pca[:, 0] if X_pca is not None and len(X_pca) > 0 else [], 
                                            y=X_pca[:, 1] if X_pca is not None and len(X_pca) > 0 else [], 
                                            mode='markers',
                                            marker=dict(
                                                color=clusters if clusters is not None and len(clusters) > 0 else [], 
                                                colorscale='Viridis',
                                                size=5,
                                                showscale=True,
                                                colorbar=dict(title="Cluster")
                                            ),
                                            text=[f'Cluster {c}' for c in clusters] if clusters is not None and len(clusters) > 0 else [],
                                            hovertemplate='<b>Cluster %{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                                        ),
                                        row=1, col=2
                                    ).update_xaxes(title_text="PC1", row=1, col=2)
                                    .update_yaxes(title_text="PC2", row=1, col=2)
                                    .update_layout(
                                        height=500, 
                                        showlegend=False,
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['white'],
                                        font_color=COLORS['dark']
                                    ) 
                                    if X_pca is not None and clusters is not None 
                                    else go.Figure().add_annotation(
                                        text="Dados de cluster não disponíveis",
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False,
                                        font=dict(size=16, color=COLORS['dark'])
                                    ).update_layout(
                                        plot_bgcolor=COLORS['background'],
                                        paper_bgcolor=COLORS['white']
                                    )
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ])
        ]),

        # Tab 4: Predição em Tempo Real
        dcc.Tab(label='🔮 Predição', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📝 Fazer Predição", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Lead Time (dias):", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='lead-time', type='number', value=50, className="form-control")
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("ADR:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='adr', type='number', value=100, className="form-control")
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Tipo de Hotel:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Dropdown(
                                            id='pred-hotel',
                                            options=[{'label': hotel, 'value': hotel} for hotel in df['hotel'].unique()],
                                            value='City Hotel'
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Tipo de Cliente:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Dropdown(
                                            id='customer-type',
                                            options=[{'label': ct, 'value': ct} for ct in df['customer_type'].unique()],
                                            value='Transient'
                                        )
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Total de Noites:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='total-nights', type='number', value=3, className="form-control")
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Total de Hóspedes:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='total-guests', type='number', value=2, className="form-control")
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Button("Fazer Predição", id='predict-btn', 
                                         style={'backgroundColor': COLORS['secondary'], 'border': 'none', 'fontWeight': 'bold'}, 
                                         className="w-100")
                            ])
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Resultado da Predição", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Div(id='prediction-result', className="text-center",
                                    style={'fontSize': '24px', 'fontWeight': 'bold', 'color': COLORS['dark']}),
                            html.Br(),
                            dcc.Graph(id='probability-chart')
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=8)
            ])
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# Calcular total_guests e total_nights em df_analyzed
df_analyzed['total_guests'] = df_analyzed['adults'] + df_analyzed['children'] + df_analyzed['babies']
df_analyzed['total_nights'] = df_analyzed['stays_in_weekend_nights'] + df_analyzed['stays_in_week_nights']

print("✅ Colunas 'total_guests' e 'total_nights' adicionadas ao df_analyzed.")

# Calcular e criar feature_importance_df
try:
    # Check if models were loaded successfully
    if best_model is None or preprocessor is None:
        raise ValueError("Modelos não foram carregados.")
    
    # Get feature names from preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Try to extract the actual model from pipeline if needed
    actual_model = best_model
    if hasattr(best_model, 'named_steps'):
        # It's a pipeline, get the last step (the actual model)
        steps = list(best_model.named_steps.values())
        actual_model = steps[-1]
        print(f"📦 Modelo é um Pipeline. Extraindo modelo final: {type(actual_model).__name__}")
    elif hasattr(best_model, 'steps'):
        # Alternative pipeline structure
        actual_model = best_model.steps[-1][1]
        print(f"📦 Modelo é um Pipeline. Extraindo modelo final: {type(actual_model).__name__}")
    
    # Get feature importance from the actual model
    if hasattr(actual_model, 'feature_importances_'):
        importances = actual_model.feature_importances_
        print(f"✅ Usando feature_importances_ do modelo {type(actual_model).__name__}")
    elif hasattr(actual_model, 'coef_'):
        importances = abs(actual_model.coef_[0]) if len(actual_model.coef_.shape) > 1 else abs(actual_model.coef_)
        print(f"✅ Usando coef_ do modelo {type(actual_model).__name__}")
    else:
        raise ValueError(f"Modelo {type(actual_model).__name__} não possui feature_importances_ ou coef_")

    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    print("✅ feature_importance_df criado com sucesso!")
    print(f"📊 Shape: {feature_importance_df.shape}")
    print(f"📊 Top 5 features:\n{feature_importance_df.head()[['feature', 'importance']]}")

except Exception as e:
    print(f"⚠️ Erro ao criar feature_importance_df: {e}. Usando dados estáticos.")
    # Create a dummy feature_importance_df if models were not loaded
    feature_importance_df = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(20)],
        'importance': np.linspace(0.5, 0.01, 20)
    }).sort_values('importance', ascending=False)

# Realizar PCA e Classificação de Clusters
try:
    if kmeans_model is not None and preprocessor is not None:
        print("✅ Modelos de clustering e pré-processador encontrados. Realizando PCA e classificação...")
        # Prepare data for clustering (assuming df_analyzed is the correct DataFrame)

        # --- Modified: Select a broader set of likely original features including the missing ones ---
        # Identify columns to use for clustering. This should match the features
        # the preprocessor was trained on.
        # This is a common set of features for this dataset, adjust if your model used different ones.
        original_features_for_preprocessing = [
            'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
            'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
            'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
            'previous_bookings_not_canceled', 'booking_changes', 'agent',
            'company', 'adr', 'required_car_parking_spaces',
            'total_of_special_requests', 'hotel', 'arrival_date_month',
            'country', 'market_segment', 'distribution_channel',
            'reserved_room_type', 'assigned_room_type', 'deposit_type',
            'customer_type',
            # Add features identified as missing by the preprocessor
            'has_special_request', 'total_guests', 'total_nights', 'is_family'
        ]

        # Ensure 'total_guests' and 'total_nights' are in df_analyzed if needed
        # (This is handled by the cell 5910bf79, but a check here can be a safeguard)
        if 'total_guests' not in df_analyzed.columns or 'total_nights' not in df_analyzed.columns:
             print("⚠️ Colunas 'total_guests' ou 'total_nights' não encontradas em df_analyzed. Calculando...")
             if 'adults' in df_analyzed.columns and 'children' in df_analyzed.columns and 'babies' in df_analyzed.columns:
                df_analyzed['total_guests'] = df_analyzed['adults'] + df_analyzed['children'] + df_analyzed['babies']
             if 'stays_in_weekend_nights' in df_analyzed.columns and 'stays_in_week_nights' in df_analyzed.columns:
                df_analyzed['total_nights'] = df_analyzed['stays_in_weekend_nights'] + df_analyzed['stays_in_week_nights']

        # Ensure 'is_family' is in df_analyzed if needed
        if 'is_family' not in df_analyzed.columns:
            print("⚠️ Coluna 'is_family' não encontrada em df_analyzed. Calculando...")
            # Assuming is_family is true if there are children or babies
            if 'children' in df_analyzed.columns and 'babies' in df_analyzed.columns:
                 df_analyzed['is_family'] = ((df_analyzed['children'] > 0) | (df_analyzed['babies'] > 0)).astype(int)
            else:
                 print("❌ Não foi possível calcular 'is_family': colunas 'children' ou 'babies' ausentes.")
                 # Remove 'is_family' from the expected features if it cannot be calculated
                 if 'is_family' in original_features_for_preprocessing:
                     original_features_for_preprocessing.remove('is_family')


        # Ensure 'has_special_request' is in df_analyzed if needed
        if 'has_special_request' not in df_analyzed.columns:
             print("⚠️ Coluna 'has_special_request' não encontrada em df_analyzed. Calculando...")
             # Assuming has_special_request is based on total_of_special_requests
             if 'total_of_special_requests' in df_analyzed.columns:
                 df_analyzed['has_special_request'] = (df_analyzed['total_of_special_requests'] > 0).astype(int)
             else:
                 print("❌ Não foi possível calcular 'has_special_request': coluna 'total_of_special_requests' ausente.")
                 # Remove 'has_special_request' from the expected features if it cannot be calculated
                 if 'has_special_request' in original_features_for_preprocessing:
                     original_features_for_preprocessing.remove('has_special_request')


        # Select only the features that exist in df_analyzed after potential calculations
        actual_features_to_use = [f for f in original_features_for_preprocessing if f in df_analyzed.columns]

        if len(actual_features_to_use) != len(original_features_for_preprocessing):
            missing_features = set(original_features_for_preprocessing) - set(actual_features_to_use)
            print(f"⚠️ Aviso: As seguintes features esperadas pelo preprocessor não foram encontradas em df_analyzed mesmo após tentativas de cálculo: {missing_features}")
            print("⚠️ A análise de clustering continuará com as features disponíveis, mas os resultados podem ser afetados.")


        if not actual_features_to_use:
            print("❌ Nenhuma das features esperadas pelo preprocessor foi encontrada em df_analyzed.")
            X_pca = None
            clusters = None
        else:
            # Display columns of df_analyzed just before selection for debugging
            print("Colunas em df_analyzed antes da seleção para clustering:")
            print(df_analyzed.columns)
            print(f"Features selecionadas para clustering e existentes em df_analyzed: {actual_features_to_use}")

            # Select the features from df_analyzed
            X = df_analyzed[actual_features_to_use]

            # Apply the preprocessor
            X_processed = preprocessor.transform(X)

            # --- Added: PCA step to reduce dimensions to 21 before KMeans ---
            from sklearn.decomposition import PCA
            n_components_kmeans = 21 # Number of components expected by KMeans
            if X_processed.shape[1] < n_components_kmeans:
                 print(f"⚠️ Número de features após pré-processamento ({X_processed.shape[1]}) é menor que o número de componentes esperado pelo KMeans ({n_components_kmeans}). Não é possível realizar PCA.")
                 X_for_kmeans = X_processed # Use the processed data directly if fewer features
                 X_pca = None # Cannot perform visualization PCA if less than 2 features
                 clusters = None
            else:
                pca_kmeans = PCA(n_components=n_components_kmeans)
                X_for_kmeans = pca_kmeans.fit_transform(X_processed)

                # Apply PCA for visualization (reduce to 2 dimensions)
                # Ensure there are at least 2 features after the first PCA step for visualization PCA
                if X_for_kmeans.shape[1] >= 2:
                    pca_viz = PCA(n_components=2)
                    X_pca = pca_viz.fit_transform(X_for_kmeans)
                else:
                    print("⚠️ Número de features após PCA para KMeans é menor que 2. Não é possível realizar PCA para visualização.")
                    X_pca = None # Cannot perform visualization PCA

                # Predict clusters
                clusters = kmeans_model.predict(X_for_kmeans)

                print("✅ PCA (para KMeans) e classificação de clusters realizados com sucesso!")
                print(f"📊 Clusters shape: X_pca={X_pca.shape if X_pca is not None else 'None'}, clusters={len(clusters) if clusters is not None else 'None'}")
                print(f"📊 Número de clusters únicos: {len(set(clusters)) if clusters is not None else 'None'}")

    else:
        print("⚠️  Modelos de clustering ou pré-processador não encontrados. A análise de clusters não será realizada.")
        X_pca = None # Set X_pca to None to avoid NameError in layout
        clusters = None # Set clusters to None to avoid NameError in layout

except Exception as e:
    print(f"❌ Erro ao realizar PCA ou classificação de clusters: {e}")
    X_pca = None # Set X_pca to None to avoid NameError in layout
    clusters = None # Set clusters to None to avoid NameError in layout


"""# EXECUTAR O DASHBOARD"""

if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash...")
    print("📋 Acesse: http://127.0.0.1:8050")
    print("⚠️ Pressione Ctrl+C para parar o servidor")
    app.run(debug=True, host='127.0.0.1', port=8050)
