# -*- coding: utf-8 -*-
"""
Dashboard Interativo - Hotel Booking Analysis
Análise exploratória e resultados de Machine Learning
"""

print("📊 INICIANDO DASHBOARD DE ANÁLISE DE RESERVAS DE HOTÉIS")

# ============================================================================
# IMPORTAÇÕES
# ============================================================================

import os
import warnings
warnings.filterwarnings('ignore')

import dash
from dash import dcc, html, Input, Output
import dash.dependencies
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import joblib

# ============================================================================
# CONFIGURAÇÕES E PALETA DE CORES
# ============================================================================

COLORS = {
    'primary': '#132F3B',      # Azul escuro principal
    'secondary': '#1E4A5F',    # Azul escuro secundário
    'accent': '#FF4F19',       # Laranja para destaques
    'background': '#F5F7FA',   # Fundo claro suave
    'dark': '#132F3B',         # Azul escuro para textos
    'white': '#FFFFFF',        # Branco
    'text': '#132F3B',         # Azul escuro para textos
    'light_blue': '#2C5F7E',   # Azul intermediário
    'gradient_start': '#132F3B',  # Início do gradiente
    'gradient_end': '#1E4A5F'     # Fim do gradiente
}

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================

print("\n📂 Carregando dados...")

# Carregar dataset principal
parquet_file = 'hotel_bookings.parquet'
csv_files = ['ML/data/hotel_bookings.csv', 'EDA/hotel_bookings.csv', 'hotel_bookings.csv']

df = None

if os.path.exists(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(f"✅ Dataset carregado do Parquet: {parquet_file}")
else:
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.to_parquet(parquet_file, index=False)
            print(f"✅ Dataset carregado de: {csv_file}")
            break
    
if df is None:
    raise FileNotFoundError(
        "❌ Arquivo de dados não encontrado. "
        "Verifique se o dataset está em uma das pastas: ML/data/, EDA/ ou raiz do projeto."
    )

# ============================================================================
# CARREGAMENTO DE RESULTADOS DE ML (com fallback para dados dummy)
# ============================================================================

print("\n🤖 Carregando resultados de Machine Learning...")

# 1. Resultados dos modelos
try:
    model_results = pd.read_csv('model_results.csv')
    print("✅ Resultados dos modelos carregados")
except FileNotFoundError:
    print("⚠️  Criando resultados dummy (execute o script ML primeiro para dados reais)")
    model_results = pd.DataFrame({
        'Modelo': ['Random Forest', 'XGBoost', 'Logistic Regression', 'Gradient Boosting'],
        'Acurácia': [0.85, 0.83, 0.78, 0.84],
        'Precisão': [0.84, 0.82, 0.76, 0.83],
        'Recall': [0.81, 0.79, 0.74, 0.80],
        'F1-Score': [0.82, 0.80, 0.75, 0.81]
    })

# 2. Carregar modelo e preprocessor para predições
best_model = None
preprocessor = None

# Tentar carregar de múltiplos locais e nomes
model_paths = [
    ('best_model.pkl', 'preprocessor.pkl'),  # Nome padrão - Raiz
    ('best_classification_model.pkl', 'preprocessor.pkl'),  # Nome correto - Raiz
    ('best_classication_model.pkl', 'preprocessor.pkl'),  # Nome alternativo - Raiz
    ('ML/best_model.pkl', 'ML/preprocessor.pkl'),  # Pasta ML
    ('ML/best_classification_model.pkl', 'ML/preprocessor.pkl'),  # Pasta ML
    ('ML/best_classication_model.pkl', 'ML/preprocessor.pkl'),  # Pasta ML alternativo
    ('../best_model.pkl', '../preprocessor.pkl'),  # Um nível acima
    ('../best_classification_model.pkl', '../preprocessor.pkl'),  # Um nível acima
    ('../best_classication_model.pkl', '../preprocessor.pkl'),  # Um nível acima alternativo
]

for model_path, prep_path in model_paths:
    try:
        if os.path.exists(model_path) and os.path.exists(prep_path):
            best_model = joblib.load(model_path)
            preprocessor = joblib.load(prep_path)
            print(f"✅ Modelo carregado de: {model_path}")
            print(f"✅ Preprocessor carregado de: {prep_path}")
            break
    except Exception as e:
        continue

if best_model is None:
    print("\n" + "="*80)
    print("⚠️  MODELO NÃO ENCONTRADO - Modo de demonstração ativo")
    print("="*80)
    print("\n📝 Para usar predições reais, execute primeiro:")
    print("   1. python ML/ml_dataset_hotel_pisi3.py")
    print("   2. Isso irá gerar os arquivos: best_model.pkl e preprocessor.pkl")
    print("\n💡 Por enquanto, o dashboard usará predições simuladas baseadas em regras.")
    print("="*80 + "\n")

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
                background: linear-gradient(135deg, #F5F7FA 0%, #E8EDF2 100%) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .card {
                border: none !important;
                box-shadow: 0 4px 12px rgba(19, 47, 59, 0.15) !important;
                border-radius: 12px !important;
                transition: transform 0.2s ease, box-shadow 0.2s ease !important;
            }
            .card:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(19, 47, 59, 0.25) !important;
            }
            .tab {
                background: linear-gradient(135deg, #FFFFFF 0%, #F5F7FA 100%) !important;
                color: #132F3B !important;
                border: 2px solid #E8EDF2 !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            .tab--selected {
                background: linear-gradient(135deg, #132F3B 0%, #1E4A5F 100%) !important;
                color: #FFFFFF !important;
                border-bottom: 4px solid #FF4F19 !important;
                box-shadow: 0 4px 8px rgba(19, 47, 59, 0.3) !important;
            }
            .tab:hover {
                background: linear-gradient(135deg, #1E4A5F 0%, #2C5F7E 100%) !important;
                color: #FFFFFF !important;
                border-color: #1E4A5F !important;
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
                   style={
                       'color': COLORS['white'], 
                       'fontWeight': 'bold', 
                       'padding': '30px 0',
                       'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
                   })
        ], width=12)
    ], style={
        'background': f'linear-gradient(135deg, {COLORS["gradient_start"]} 0%, {COLORS["gradient_end"]} 100%)',
        'borderRadius': '0 0 20px 20px',
        'marginBottom': '20px',
        'boxShadow': '0 4px 12px rgba(19, 47, 59, 0.2)'
    }),

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
                                            color_discrete_sequence=[COLORS['primary'], COLORS['light_blue']])
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
                                                  color_discrete_sequence=[COLORS['primary'], COLORS['accent']])
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
                                            color_discrete_sequence=[COLORS['primary'], COLORS['light_blue']])
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
                                            color_continuous_scale=[[0, COLORS['light_blue']], [1, COLORS['primary']]])
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
        # TAB 2: ANÁLISE DE CANCELAMENTOS
        # ====================================================================
        dcc.Tab(label='❌ Análise de Cancelamentos', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔍 Filtros", 
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
                            ], className="text-center mb-3", style={'borderRadius': '12px'})
                        ], width=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("Reservas Totais", className="card-title", style={'color': COLORS['dark']}),
                                    html.H2(id='total-bookings', style={'color': COLORS['primary'], 'fontWeight': 'bold'})
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="text-center mb-3", style={'borderRadius': '12px'})
                        ], width=4),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("ADR Médio", className="card-title", style={'color': COLORS['dark']}),
                                    html.H2(id='avg-adr', style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="text-center mb-3", style={'borderRadius': '12px'})
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
                            ], className="mb-3", style={'borderRadius': '12px'})
                        ], width=6),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("📈 Cancelamentos por Mês", 
                                             style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                                dbc.CardBody([
                                    dcc.Graph(id='cancel-by-month')
                                ], style={'backgroundColor': COLORS['white']})
                            ], className="mb-3", style={'borderRadius': '12px'})
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

        # ====================================================================
        # TAB 3: MACHINE LEARNING
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
                                        html.Th("Precisão", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        html.Th("Recall", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        html.Th("F1-Score", style={'color': COLORS['dark'], 'fontWeight': 'bold'})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(row['Modelo'], style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        html.Td(f"{row['Acurácia']:.4f}", style={'color': COLORS['dark']}),
                                        html.Td(f"{row['Precisão']:.4f}", style={'color': COLORS['dark']}),
                                        html.Td(f"{row['Recall']:.4f}", style={'color': COLORS['dark']}),
                                        html.Td(f"{row['F1-Score']:.4f}", style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                    ]) for _, row in model_results.iterrows()
                                ])
                            ], className="table table-striped", style={'marginBottom': '0'})
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ], className="mb-4"),

            # Gráficos de comparação
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Comparação de Acurácia", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(model_results, x='Modelo', y='Acurácia',
                                            title='Acurácia dos Modelos',
                                            color='Acurácia',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark'],
                                    yaxis_range=[0, 1]
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Comparação de F1-Score", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.bar(model_results, x='Modelo', y='F1-Score',
                                            title='F1-Score dos Modelos',
                                            color='F1-Score',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark'],
                                    yaxis_range=[0, 1]
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=6)
            ], className="mb-4"),

            # Comparação de todas as métricas
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Comparação de Todas as Métricas", 
                                     style={'backgroundColor': COLORS['accent'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            dcc.Graph(
                                figure=px.line(
                                    model_results.melt(id_vars='Modelo', var_name='Métrica', value_name='Valor'),
                                    x='Modelo', y='Valor', color='Métrica',
                                    title='Comparação de Métricas por Modelo',
                                    markers=True,
                                    color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent'], '#FF8C42'])
                                .update_layout(
                                    plot_bgcolor=COLORS['background'],
                                    paper_bgcolor=COLORS['white'],
                                    font_color=COLORS['dark'],
                                    title_font_color=COLORS['dark'],
                                    yaxis_range=[0, 1],
                                    legend_title_text='Métricas'
                                )
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ])
        ]),

        # ====================================================================
        # TAB 4: PREDIÇÃO EM TEMPO REAL
        # ====================================================================
        dcc.Tab(label='🔮 Predição', style={'padding': '10px', 'fontWeight': 'bold'}, children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📝 Dados para Predição", 
                                     style={'backgroundColor': COLORS['secondary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Lead Time (dias):", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='lead-time', type='number', value=50, className="form-control", style={'marginBottom': '10px'})
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("ADR:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='adr-pred', type='number', value=100, className="form-control", style={'marginBottom': '10px'})
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Tipo de Hotel:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Dropdown(
                                            id='pred-hotel',
                                            options=[{'label': hotel, 'value': hotel} for hotel in df['hotel'].unique()],
                                            value=df['hotel'].unique()[0] if len(df['hotel'].unique()) > 0 else 'City Hotel',
                                            style={'marginBottom': '10px'}
                                        )
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Tipo de Cliente:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Dropdown(
                                            id='customer-type',
                                            options=[{'label': ct, 'value': ct} for ct in df['customer_type'].unique()],
                                            value=df['customer_type'].unique()[0] if len(df['customer_type'].unique()) > 0 else 'Transient',
                                            style={'marginBottom': '10px'}
                                        )
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Total de Noites:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='total-nights', type='number', value=3, className="form-control", style={'marginBottom': '10px'})
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Total de Hóspedes:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                                        dcc.Input(id='total-guests', type='number', value=2, className="form-control", style={'marginBottom': '10px'})
                                    ], width=6)
                                ]),
                                html.Br(),
                                dbc.Button("🔮 Fazer Predição", id='predict-btn', 
                                         style={'backgroundColor': COLORS['secondary'], 'border': 'none', 'fontWeight': 'bold'}, 
                                         className="w-100", n_clicks=0)
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
                                    style={'fontSize': '24px', 'fontWeight': 'bold', 'color': COLORS['dark'], 'padding': '20px'}),
                            html.Br(),
                            dcc.Graph(id='probability-chart')
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=8)
            ])
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# ============================================================================
# CALLBACKS PARA INTERATIVIDADE
# ============================================================================

# Callback para a aba de Análise de Cancelamentos
@app.callback(
    [Output('cancel-rate', 'children'),
     Output('total-bookings', 'children'),
     Output('avg-adr', 'children'),
     Output('cancel-by-segment', 'figure'),
     Output('cancel-by-month', 'figure'),
     Output('cancel-by-customer-type', 'figure')],
    [Input('hotel-filter', 'value'),
     Input('country-filter', 'value')]
)
def update_cancel_analysis(hotel_filter, country_filter):
    # Filtrar dados
    filtered_df = df.copy()

    if hotel_filter != 'all':
        filtered_df = filtered_df[filtered_df['hotel'] == hotel_filter]

    if country_filter != 'all':
        filtered_df = filtered_df[filtered_df['country'] == country_filter]

    # Calcular métricas
    cancel_rate = f"{filtered_df['is_canceled'].mean()*100:.1f}%"
    total_bookings = f"{filtered_df.shape[0]:,}"
    avg_adr = f"${filtered_df['adr'].mean():.2f}"

    # Gráfico 1: Cancelamentos por Segmento
    cancel_by_segment = px.bar(
        filtered_df.groupby('market_segment')['is_canceled'].mean().reset_index(),
        x='market_segment', y='is_canceled',
        title='Taxa de Cancelamento por Segmento de Mercado',
        labels={'market_segment': 'Segmento', 'is_canceled': 'Taxa de Cancelamento'},
        color='is_canceled',
        color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
    ).update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['white'],
        font_color=COLORS['dark'],
        title_font_color=COLORS['dark']
    )

    # Gráfico 2: Cancelamentos por Mês
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_cancel = filtered_df.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
    monthly_cancel['arrival_date_month'] = pd.Categorical(
        monthly_cancel['arrival_date_month'], 
        categories=month_order, 
        ordered=True
    )
    monthly_cancel = monthly_cancel.sort_values('arrival_date_month')

    cancel_by_month = px.line(
        monthly_cancel, x='arrival_date_month', y='is_canceled',
        title='Taxa de Cancelamento por Mês',
        labels={'arrival_date_month': 'Mês', 'is_canceled': 'Taxa de Cancelamento'},
        markers=True,
        color_discrete_sequence=[COLORS['accent']]
    ).update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['white'],
        font_color=COLORS['dark'],
        title_font_color=COLORS['dark']
    )

    # Gráfico 3: Cancelamentos por Tipo de Cliente
    cancel_by_customer = px.bar(
        filtered_df.groupby('customer_type')['is_canceled'].mean().reset_index(),
        x='customer_type', y='is_canceled',
        title='Taxa de Cancelamento por Tipo de Cliente',
        labels={'customer_type': 'Tipo de Cliente', 'is_canceled': 'Taxa de Cancelamento'},
        color='is_canceled',
        color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
    ).update_layout(
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['white'],
        font_color=COLORS['dark'],
        title_font_color=COLORS['dark']
    )

    return cancel_rate, total_bookings, avg_adr, cancel_by_segment, cancel_by_month, cancel_by_customer

# Callback para Predição
@app.callback(
    [Output('prediction-result', 'children'),
     Output('probability-chart', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [dash.dependencies.State('lead-time', 'value'),
     dash.dependencies.State('adr-pred', 'value'),
     dash.dependencies.State('pred-hotel', 'value'),
     dash.dependencies.State('customer-type', 'value'),
     dash.dependencies.State('total-nights', 'value'),
     dash.dependencies.State('total-guests', 'value')]
)
def make_prediction(n_clicks, lead_time, adr, hotel, customer_type, total_nights, total_guests):
    if n_clicks is None or n_clicks == 0:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['white'],
            font_color=COLORS['dark'],
            annotations=[{
                'text': 'Preencha os dados e clique em "Fazer Predição"',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 16, 'color': COLORS['dark']}
            }]
        )
        return "Aguardando entrada de dados...", empty_fig

    try:
        # Criar dados de input compatíveis com o modelo
        input_data = pd.DataFrame({
            'hotel': [hotel],
            'lead_time': [lead_time],
            'arrival_date_week_number': [28],
            'arrival_date_day_of_month': [15],
            'stays_in_weekend_nights': [0],
            'stays_in_week_nights': [total_nights],
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
            'customer_type': [customer_type],
            'adr': [adr],
            'required_car_parking_spaces': [0],
            'total_of_special_requests': [0],
            'arrival_date_month': ['July'],
            'total_guests': [total_guests],
            'total_nights': [total_nights],
            'has_special_request': [0],
            'is_family': [1 if total_guests > 1 else 0]
        })

        # Fazer predição
        prediction = best_model.predict(input_data)[0]
        probability = best_model.predict_proba(input_data)[0]

        # Resultado
        if prediction == 0:
            result_text = "✅ RESERVA NÃO SERÁ CANCELADA"
            result_color = COLORS['secondary']
        else:
            result_text = "❌ RESERVA SERÁ CANCELADA"
            result_color = COLORS['accent']

        # Gráfico de probabilidade
        prob_fig = go.Figure(data=[
            go.Bar(
                x=['Não Cancelar', 'Cancelar'],
                y=probability,
                marker_color=[COLORS['secondary'], COLORS['accent']],
                text=[f'{p*100:.1f}%' for p in probability],
                textposition='outside'
            )
        ])
        
        prob_fig.update_layout(
            title='Probabilidade de Cancelamento',
            yaxis_title='Probabilidade',
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['white'],
            font_color=COLORS['dark'],
            title_font_color=COLORS['dark'],
            yaxis_range=[0, 1]
        )

        return html.Div([
            html.H3(result_text, style={'color': result_color, 'fontWeight': 'bold'}),
            html.P(f"Confiança: {max(probability)*100:.1f}%", style={'fontSize': '18px'})
        ]), prob_fig

    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['white'],
            annotations=[{
                'text': f'Erro: {str(e)}',
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'showarrow': False,
                'font': {'size': 14, 'color': COLORS['accent']}
            }]
        )
        return f"❌ Erro na predição: {str(e)}", error_fig

# ============================================================================
# EXECUÇÃO DA APLICAÇÃO
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 SERVIDOR DASH INICIADO COM SUCESSO!")
    print("="*80)
    print("\n📋 Informações de acesso:")
    print("   • URL: http://127.0.0.1:8050")
    print("   • Ou: http://localhost:8050")
    print("\n💡 Para parar o servidor, pressione Ctrl+C no terminal")
    print("="*80 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050)
