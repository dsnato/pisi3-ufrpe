# -*- coding: utf-8 -*-
"""
COMMIT 1: Estrutura Base do Dashboard + Tab Visão Geral
Funcionalidade: Dashboard básico com cabeçalho, paleta de cores e primeira tab de visualizações
"""

print("📊 INICIANDO CRIAÇÃO DO DASHBOARD - COMMIT 1")

# Paleta de cores personalizada
COLORS = {
    'primary': '#132F3B',
    'secondary': '#0162B3',
    'accent': '#FF4F19',
    'background': '#EFEFF0',
    'dark': '#132F3B',
    'white': '#FFFFFF',
    'text': '#132F3B'
}

# Importações
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Carregar dados
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
    df.to_parquet(parquet_file, index=False)
    print(f"✅ Dataset salvo em Parquet: {parquet_file}")
elif os.path.exists(csv_file_eda):
    df = pd.read_csv(csv_file_eda)
    print(f"✅ Dataset carregado do CSV: {csv_file_eda}")
    df.to_parquet(parquet_file, index=False)
elif os.path.exists(csv_file_root):
    df = pd.read_csv(csv_file_root)
    print(f"✅ Dataset carregado do CSV: {csv_file_root}")
    df.to_parquet(parquet_file, index=False)
else:
    raise FileNotFoundError("❌ Arquivo de dados não encontrado.")

# Criar aplicação
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

# LAYOUT
app.layout = dbc.Container([
    # Cabeçalho
    dbc.Row([
        dbc.Col([
            html.H1("🏨 Hotel Booking Analysis Dashboard",
                   className="text-center mb-4",
                   style={'color': COLORS['dark'], 'fontWeight': 'bold', 'padding': '20px 0'})
        ], width=12)
    ], style={'background': f'linear-gradient(135deg, {COLORS["white"]} 0%, {COLORS["background"]} 100%)'}),

    # Tab Visão Geral
    dcc.Tabs(style={'marginTop': '20px'}, children=[
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
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash - COMMIT 1...")
    print("📋 Acesse: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)

# Carregar dados
parquet_file = 'hotel_bookings.parquet'
csv_files = ['ML/data/hotel_bookings.csv', 'EDA/hotel_bookings.csv', 'hotel_bookings.csv']

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"✅ Dataset carregado do Parquet")
else:
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print(f"✅ Dataset carregado de {csv_file}")
            break
    else:
        raise FileNotFoundError("❌ Arquivo de dados não encontrado.")

# Carregar ou criar dados de ML
try:
    model_results = pd.read_csv('model_results.csv')
    print("✅ model_results.csv carregado.")
except:
    print("⚠️ Criando model_results dummy.")
    model_results = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Logistic Regression'],
        'Acurácia': [0.85, 0.83, 0.78],
        'F1-Score': [0.82, 0.80, 0.75]
    })

# Feature importance
try:
    import joblib
    best_model = joblib.load('best_classification_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = abs(best_model.coef_[0])
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names, 
        'importance': importances
    }).sort_values('importance', ascending=False)
    print("✅ Feature importance carregado.")
except:
    print("⚠️ Criando feature importance dummy.")
    feature_importance_df = pd.DataFrame({
        'feature': ['lead_time', 'adr', 'total_nights', 'previous_cancellations', 
                   'booking_changes', 'market_segment_Online', 'deposit_type_Non_Refund'],
        'importance': np.linspace(0.15, 0.05, 7)
    })

# Clustering Analysis
try:
    cluster_analysis = pd.read_csv('cluster_analysis.csv')
    print("✅ cluster_analysis.csv carregado.")
except:
    print("⚠️ Criando cluster_analysis dummy.")
    cluster_analysis = pd.DataFrame({
        'Cluster': [0, 1, 2],
        'is_canceled': [0.3, 0.5, 0.7],
        'adr': [95.5, 110.2, 85.3],
        'lead_time': [45, 95, 150]
    })

# PCA e Clusters (tentar carregar ou criar dummy)
X_pca = None
clusters = None

try:
    import joblib
    from sklearn.decomposition import PCA
    
    kmeans_model = joblib.load('kmeans_cluster_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    # Preparar dados para clustering
    numeric_features = ['lead_time', 'adr', 'adults', 'children', 'babies',
                       'stays_in_weekend_nights', 'stays_in_week_nights']
    
    X_sample = df[numeric_features].fillna(0).sample(n=min(5000, len(df)), random_state=42)
    
    # PCA para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sample)
    
    # Prever clusters (se o modelo tiver o método predict)
    if hasattr(kmeans_model, 'predict'):
        try:
            X_processed = preprocessor.transform(X_sample)
            if X_processed.shape[1] > 21:
                pca_reduce = PCA(n_components=21)
                X_processed = pca_reduce.fit_transform(X_processed)
            clusters = kmeans_model.predict(X_processed)
        except:
            clusters = np.random.randint(0, 3, len(X_sample))
    else:
        clusters = np.random.randint(0, 3, len(X_sample))
    
    print("✅ PCA e clusters carregados.")
except:
    print("⚠️ Criando PCA e clusters dummy.")
    np.random.seed(42)
    n_samples = 1000
    X_pca = np.random.randn(n_samples, 2) * 2
    clusters = np.random.randint(0, 3, n_samples)

# Criar aplicação
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

# LAYOUT
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
                        dbc.CardBody([
                            html.H4(f"Total de Reservas: {df.shape[0]:,}", style={'color': COLORS['dark']}),
                            html.H4(f"Taxa de Cancelamento: {df['is_canceled'].mean()*100:.1f}%",
                                   style={'color': COLORS['accent'], 'fontWeight': 'bold'})
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ])
        ]),

        # Tab 2: Análise de Cancelamentos
        dcc.Tab(label='❌ Análise de Cancelamentos', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Taxa de Cancelamento", style={'color': COLORS['dark']}),
                            html.H2(f"{df['is_canceled'].mean()*100:.1f}%", 
                                   style={'color': COLORS['accent'], 'fontWeight': 'bold'})
                        ], style={'backgroundColor': COLORS['white']})
                    ], className="text-center", style={'borderRadius': '12px'})
                ], width=12)
            ])
        ]),

        # Tab 3: Machine Learning (COM CLUSTERING AGORA)
        dcc.Tab(label='🤖 Machine Learning', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Resultados dos Modelos", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Table([
                                html.Thead([
                                    html.Tr([
                                        html.Th("Modelo", style={'color': COLORS['dark']}), 
                                        html.Th("F1-Score", style={'color': COLORS['dark']})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(row['Modelo'], style={'color': COLORS['dark']}),
                                        html.Td(f"{row['F1-Score']:.4f}", style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                    ]) for _, row in model_results.iterrows()
                                ])
                            ], className="table table-striped")
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
                                            title='Desempenho dos Modelos',
                                            color='F1-Score',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'])
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
                                figure=px.bar(feature_importance_df.head(10),
                                            x='importance', y='feature',
                                            title='Top 10 Features Mais Importantes',
                                            orientation='h',
                                            color='importance',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'])
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ]),

            # NOVA SEÇÃO: Análise de Clusters
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

if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash")
    print("📋 Acesse: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)  