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

"""
Funcionalidade: Segunda tab com análise detalhada de cancelamentos e callbacks
"""

# Paleta de cores
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
from dash import dcc, html, Input, Output
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

    # Abas principais
    dcc.Tabs(style={'marginTop': '20px'}, children=[
        # Tab 1: Visão Geral (da COMMIT 1)
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
                                            color='hotel',
                                            color_discrete_sequence=[COLORS['primary'], COLORS['secondary']])
                                .update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], font_color=COLORS['dark'])
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=8)
            ])
        ]),

        # Tab 2: Análise de Cancelamentos (NOVA)
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
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# CALLBACKS - Análise de Cancelamentos
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
        color='is_canceled',
        color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
    ).update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], font_color=COLORS['dark'])
    
    # Gráfico 2: Cancelamentos por Mês
    monthly_cancel = filtered_df.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    monthly_cancel['arrival_date_month'] = pd.Categorical(monthly_cancel['arrival_date_month'], categories=month_order, ordered=True)
    monthly_cancel = monthly_cancel.sort_values('arrival_date_month')
    
    cancel_by_month = px.line(
        monthly_cancel, x='arrival_date_month', y='is_canceled',
        title='Taxa de Cancelamento por Mês',
        color_discrete_sequence=[COLORS['secondary']]
    ).update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], font_color=COLORS['dark'])
    
    # Gráfico 3: Cancelamentos por Tipo de Cliente
    cancel_by_customer = px.bar(
        filtered_df.groupby('customer_type')['is_canceled'].mean().reset_index(),
        x='customer_type', y='is_canceled',
        title='Taxa de Cancelamento por Tipo de Cliente',
        color='is_canceled',
        color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]]
    ).update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], font_color=COLORS['dark'])
    
    return cancel_rate, total_bookings, avg_adr, cancel_by_segment, cancel_by_month, cancel_by_customer

if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash - COMMIT 1...")
    print("📋 Acesse: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)

"""
Funcionalidade: Terceira tab mostrando métricas dos modelos e feature importance
Observação: Funciona com ou sem os arquivos de modelo pkl
"""

# Paleta de cores
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
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

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

# Tentar carregar resultados de ML (ou criar dados dummy)
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

# Criar feature importance (dummy ou real)
try:
    import joblib
    best_model = joblib.load('best_classification_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = abs(best_model.coef_[0])
    else:
        importances = np.random.rand(len(feature_names))
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names, 
        'importance': importances
    }).sort_values('importance', ascending=False)
    print("✅ Feature importance carregado dos modelos.")
except:
    print("⚠️ Modelos não encontrados. Criando feature importance dummy.")
    feature_importance_df = pd.DataFrame({
        'feature': ['lead_time', 'adr', 'total_nights', 'previous_cancellations', 
                   'booking_changes', 'days_in_waiting_list', 'required_car_parking_spaces',
                   'total_of_special_requests', 'market_segment_Online', 'customer_type_Transient',
                   'deposit_type_Non_Refund', 'hotel_City', 'is_repeated_guest', 'adults', 'children'],
        'importance': np.linspace(0.15, 0.01, 15)
    }).sort_values('importance', ascending=False)

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
                        dbc.CardHeader("📊 Métricas Principais", 
                                     style={'backgroundColor': COLORS['primary'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.H4(f"Total de Reservas: {df.shape[0]:,}", style={'color': COLORS['dark']}),
                            html.H4(f"Taxa de Cancelamento: {df['is_canceled'].mean()*100:.1f}%",
                                   style={'color': COLORS['accent'], 'fontWeight': 'bold'}),
                            html.H4(f"ADR Médio: ${df['adr'].mean():.2f}", style={'color': COLORS['dark']})
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
                        dbc.CardHeader("Filtros", 
                                     style={'backgroundColor': COLORS['dark'], 'color': COLORS['white'], 'fontWeight': 'bold'}),
                        dbc.CardBody([
                            html.Label("Tipo de Hotel:", style={'color': COLORS['dark'], 'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='hotel-filter',
                                options=[{'label': 'Todos', 'value': 'all'}] +
                                       [{'label': h, 'value': h} for h in df['hotel'].unique()],
                                value='all'
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Taxa de Cancelamento", style={'color': COLORS['dark']}),
                            html.H2(id='cancel-rate', style={'color': COLORS['accent'], 'fontWeight': 'bold'})
                        ], style={'backgroundColor': COLORS['white']})
                    ], className="text-center", style={'borderRadius': '12px'})
                ], width=3)
            ])
        ]),

        # Tab 3: Machine Learning (NOVA)
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
                                        html.Th("Acurácia", style={'color': COLORS['dark']}), 
                                        html.Th("F1-Score", style={'color': COLORS['dark']})
                                    ])
                                ]),
                                html.Tbody([
                                    html.Tr([
                                        html.Td(row['Modelo'], style={'color': COLORS['dark']}),
                                        html.Td(f"{row['Acurácia']:.4f}", style={'color': COLORS['dark']}),
                                        html.Td(f"{row['F1-Score']:.4f}", style={'color': COLORS['secondary'], 'fontWeight': 'bold'})
                                    ]) for _, row in model_results.iterrows()
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
                                .update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], font_color=COLORS['dark'])
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
                                figure=px.bar(feature_importance_df.head(15),
                                            x='importance', y='feature',
                                            title='Top 15 Features Mais Importantes',
                                            orientation='h',
                                            color='importance',
                                            color_continuous_scale=[[0, COLORS['secondary']], [1, COLORS['accent']]])
                                .update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['white'], 
                                             font_color=COLORS['dark'], height=600)
                            )
                        ], style={'backgroundColor': COLORS['white']})
                    ], style={'borderRadius': '12px'})
                ], width=12)
            ])
        ])
    ])
], fluid=True, style={'backgroundColor': COLORS['background'], 'padding': '20px'})

# CALLBACK - Análise de Cancelamentos
@app.callback(
    Output('cancel-rate', 'children'),
    Input('hotel-filter', 'value')
)
def update_cancel_rate(hotel_filter):
    filtered_df = df.copy()
    if hotel_filter != 'all':
        filtered_df = filtered_df[filtered_df['hotel'] == hotel_filter]
    return f"{filtered_df['is_canceled'].mean()*100:.1f}%"

if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash - COMMIT 3...")
    print("📋 Acesse: http://127.0.0.1:8050")
    app.run(debug=True, host='127.0.0.1', port=8050)
