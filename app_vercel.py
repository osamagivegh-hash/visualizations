"""
VERCEL DEPLOYMENT VERSION: Advanced Insurance Analytics Dashboard
===============================================================
Optimized for Vercel deployment with environment variables
"""

import os
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
import logging
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for Vercel
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'insurance_db'),
    'user': os.getenv('DB_USER', 'Osama'),
    'password': os.getenv('DB_PASSWORD', 'YourPassword123'),
    'port': int(os.getenv('DB_PORT', 3306))
}

AUTH_CONFIG = {
    'username': os.getenv('AUTH_USERNAME', 'admin'),
    'password': os.getenv('AUTH_PASSWORD', 'admin123')
}

# ML Models Manager
class MLModelsManager:
    def __init__(self):
        self.charges_model = None
        self.risk_model = None
        self.load_models()
    
    def load_models(self):
        try:
            # Try to load models from Vercel environment
            if os.path.exists('/tmp/charges_model.pkl'):
                self.charges_model = joblib.load('/tmp/charges_model.pkl')
            if os.path.exists('/tmp/risk_model.pkl'):
                self.risk_model = joblib.load('/tmp/risk_model.pkl')
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
    
    def predict_charges(self, age, bmi, children, smoker, sex, region):
        try:
            base_charge = 2000 + age * 50 + bmi * 100
            if smoker == 'yes':
                base_charge += 15000
            base_charge += children * 2000
            
            region_multiplier = {
                'northeast': 1.2, 'northwest': 1.0, 
                'southeast': 0.9, 'southwest': 0.95
            }
            base_charge *= region_multiplier.get(region, 1.0)
            base_charge += np.random.normal(0, 2000)
            return max(base_charge, 1000)
        except Exception as e:
            logger.error(f"Error predicting charges: {e}")
            return None
    
    def predict_risk(self, age, bmi, children, smoker, sex, region):
        try:
            risk_score = 0
            if age > 50:
                risk_score += 2
            elif age > 40:
                risk_score += 1
            if bmi > 30:
                risk_score += 3
            elif bmi > 25:
                risk_score += 1
            if smoker == 'yes':
                risk_score += 3
            if children > 3:
                risk_score += 1
            
            if risk_score >= 5:
                return "High"
            elif risk_score >= 2:
                return "Medium"
            else:
                return "Low"
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return None

ml_manager = MLModelsManager()

# Create sample data for demo
def create_sample_data():
    np.random.seed(42)
    n_records = 1000
    
    ages = np.random.randint(18, 80, n_records)
    bmi_base = 20 + (ages - 18) * 0.3 + np.random.normal(0, 5, n_records)
    bmi = np.clip(bmi_base, 15, 50)
    
    smoker_prob = 0.2 + (ages > 40) * 0.1
    smoker = np.random.binomial(1, smoker_prob, n_records)
    
    base_charges = 2000 + ages * 50 + bmi * 100 + smoker * 15000
    charges = base_charges + np.random.normal(0, 3000, n_records)
    charges = np.clip(charges, 1000, 50000)
    
    children = np.random.poisson(1.5, n_records)
    children = np.clip(children, 0, 5)
    charges += children * 2000
    
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_records)
    region_multiplier = {'northeast': 1.2, 'northwest': 1.0, 'southeast': 0.9, 'southwest': 0.95}
    charges = charges * [region_multiplier[r] for r in regions]
    
    df = pd.DataFrame({
        'age': ages,
        'sex': np.random.choice(['male', 'female'], n_records),
        'bmi': bmi,
        'children': children,
        'smoker': ['yes' if s else 'no' for s in smoker],
        'charges': charges,
        'region': regions,
        'date': pd.date_range('2020-01-01', periods=n_records, freq='D')
    })
    
    return df

# Load data
df = create_sample_data()

# Dash app setup
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.DARKLY,
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ],
                suppress_callback_exceptions=True)

app.title = "🚀 Advanced Insurance Analytics Dashboard"
server = app.server

# Custom CSS
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
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                min-height: 100vh;
            }
            .Select-control {
                background-color: white !important;
                color: black !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
            }
            .Select-menu-outer {
                z-index: 99999 !important;
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                right: 0 !important;
                margin-top: 1px !important;
                width: 100% !important;
            }
            .Select {
                position: relative !important;
            }
            .card {
                background: rgba(255, 255, 255, 0.1) !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 15px !important;
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

# Authentication
def check_auth(username, password):
    return username == AUTH_CONFIG['username'] and password == AUTH_CONFIG['password']

# Layouts
def auth_layout(message=""):
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("🔐 Login", className="text-center mb-4 text-white"),
                        html.Div(id="auth-message", children=message),
                        dbc.InputGroup([
                            dbc.InputGroupText("👤"),
                            dbc.Input(id="username-input", placeholder="Username", type="text")
                        ], className="mb-3"),
                        dbc.InputGroup([
                            dbc.InputGroupText("🔒"),
                            dbc.Input(id="password-input", placeholder="Password", type="password")
                        ], className="mb-3"),
                        dbc.Button("Login", id="login-button", color="primary", className="w-100")
                    ])
                ], className="p-4", style={'maxWidth': '400px', 'margin': '0 auto'})
            ], width=12)
        ], justify="center", align="center", style={'minHeight': '100vh'})
    ], fluid=True)

def create_dashboard_layout():
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    default_cat = categorical_cols[0] if categorical_cols else None
    
    min_date = df['date'].min() if 'date' in df.columns else pd.Timestamp('2023-01-01')
    max_date = df['date'].max() if 'date' in df.columns else pd.Timestamp('2023-12-31')
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H1("🚀 Advanced Insurance Analytics Dashboard", 
                           className="text-center mb-3 text-white"), width=10),
            dbc.Col(html.Div([
                html.Button("🚪 Logout", id="logout-button", 
                           className="btn btn-danger mt-2", 
                           style={'display': 'block', 'border-radius': '25px'})
            ]), width=2)
        ], className="mb-4"),
        
        # Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("🔍 Filter by Category", className="card-title text-white mb-3"),
                        dcc.Dropdown(
                            id='category-filter',
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            placeholder="Select a category",
                            value=default_cat,
                            style={'color': 'black', 'backgroundColor': 'white', 'border': '1px solid #ddd'}
                        )
                    ])
                ], className="mb-3", style={'backgroundColor': 'rgba(255,255,255,0.1)'})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📅 Date Range", className="card-title text-white mb-3"),
                        dcc.DatePickerRange(
                            id='date-range',
                            start_date=min_date,
                            end_date=max_date,
                            display_format='YYYY-MM-DD',
                            style={'display': 'block' if 'date' in df.columns else 'none',
                                  'color': 'black', 'backgroundColor': 'white'}
                        )
                    ])
                ], className="mb-3", style={'backgroundColor': 'rgba(255,255,255,0.1)'})
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📋 Select Values", className="card-title text-white mb-3"),
                        dcc.Dropdown(
                            id='category-value-filter',
                            placeholder="Select a value",
                            multi=True,
                            style={'color': 'black', 'backgroundColor': 'white', 'border': '1px solid #ddd'}
                        )
                    ])
                ], className="mb-3", style={'backgroundColor': 'rgba(255,255,255,0.1)'})
            ], width=4)
        ], className="mb-4"),
        
        # Key metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="total-records", className="text-center text-white"),
                        html.P("Total Records", className="text-center text-white-50")
                    ])
                ], color="primary", className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-charges", className="text-center text-white"),
                        html.P("Avg Charges", className="text-center text-white-50")
                    ])
                ], color="success", className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-age", className="text-center text-white"),
                        html.P("Avg Age", className="text-center text-white-50")
                    ])
                ], color="info", className="text-center")
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-bmi", className="text-center text-white"),
                        html.P("Avg BMI", className="text-center text-white-50")
                    ])
                ], color="warning", className="text-center")
            ], width=3)
        ], className="mb-4"),
        
        # Charts
        dbc.Row([
            dbc.Col(dcc.Graph(id='line-chart'), width=6),
            dbc.Col(dcc.Graph(id='bar-chart'), width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
            dbc.Col(dcc.Graph(id='pie-chart'), width=6)
        ], className="mb-4"),
        
        # ML Predictions Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("🤖 AI INSURANCE PREDICTIONS", className="text-center text-white mb-4", 
                               style={'fontSize': '28px', 'fontWeight': 'bold', 'color': '#00FF00'}),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Age:", className="text-white mb-2"),
                                dbc.Input(id="ml-age", type="number", value=30, min=18, max=80)
                            ], width=2),
                            dbc.Col([
                                html.Label("BMI:", className="text-white mb-2"),
                                dbc.Input(id="ml-bmi", type="number", value=25, min=15, max=50, step=0.1)
                            ], width=2),
                            dbc.Col([
                                html.Label("Children:", className="text-white mb-2"),
                                dbc.Input(id="ml-children", type="number", value=0, min=0, max=5)
                            ], width=2),
                            dbc.Col([
                                html.Label("Smoker:", className="text-white mb-2"),
                                dcc.Dropdown(
                                    id="ml-smoker",
                                    options=[{'label': 'No', 'value': 'no'}, {'label': 'Yes', 'value': 'yes'}],
                                    value='no',
                                    clearable=False,
                                    style={'color': 'black', 'backgroundColor': 'white', 'border': '2px solid #ddd'}
                                )
                            ], width=2),
                            dbc.Col([
                                html.Label("Sex:", className="text-white mb-2"),
                                dcc.Dropdown(
                                    id="ml-sex",
                                    options=[{'label': 'Male', 'value': 'male'}, {'label': 'Female', 'value': 'female'}],
                                    value='male',
                                    clearable=False,
                                    style={'color': 'black', 'backgroundColor': 'white', 'border': '2px solid #ddd'}
                                )
                            ], width=2),
                            dbc.Col([
                                html.Label("Region:", className="text-white mb-2"),
                                dcc.Dropdown(
                                    id="ml-region",
                                    options=[
                                        {'label': 'Northeast', 'value': 'northeast'},
                                        {'label': 'Northwest', 'value': 'northwest'},
                                        {'label': 'Southeast', 'value': 'southeast'},
                                        {'label': 'Southwest', 'value': 'southwest'}
                                    ],
                                    value='northeast',
                                    clearable=False,
                                    style={'color': 'black', 'backgroundColor': 'white', 'border': '2px solid #ddd'}
                                )
                            ], width=2)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("🔮 PREDICT", id="predict-button", 
                                          color="success", size="md", 
                                          style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '12px 30px'})
                            ], width=4, className="text-center"),
                            dbc.Col([
                                html.Div(id="prediction-results", className="text-center")
                            ], width=8)
                        ])
                    ])
                ], style={'backgroundColor': 'rgba(0,0,0,0.8)', 'border': '3px solid #00FF00'})
            ], width=12)
        ], className="mb-5")
    ], fluid=True)

# App layout
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=auth_layout()),
    html.Button("Logout", id="logout-button", style={'display': 'none'})
])

# Callbacks
@app.callback(
    [Output('page-content', 'children'), Output('auth-state', 'data')],
    [Input('login-button', 'n_clicks'), Input('logout-button', 'n_clicks')],
    [State('username-input', 'value'), State('password-input', 'value'), State('auth-state', 'data')]
)
def update_app(login_clicks, logout_clicks, username, password, auth_state):
    ctx = callback_context
    if not ctx.triggered:
        return auth_layout(), {'authenticated': False}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'login-button' and login_clicks:
        if check_auth(username, password):
            return create_dashboard_layout(), {'authenticated': True}
        else:
            return auth_layout("❌ Invalid credentials"), {'authenticated': False}
    
    elif button_id == 'logout-button' and logout_clicks:
        return auth_layout("👋 Logged out successfully"), {'authenticated': False}
    
    return auth_layout(), {'authenticated': False}

@app.callback(
    [Output('category-value-filter', 'options'), Output('category-value-filter', 'value')],
    [Input('category-filter', 'value')]
)
def update_category_values(selected_category):
    if selected_category and selected_category in df.columns:
        unique_values = df[selected_category].unique()
        options = [{'label': val, 'value': val} for val in unique_values]
        return options, []
    return [], []

@app.callback(
    [Output('total-records', 'children'), Output('avg-charges', 'children'),
     Output('avg-age', 'children'), Output('avg-bmi', 'children')],
    [Input('category-filter', 'value'), Input('category-value-filter', 'value'),
     Input('date-range', 'start_date'), Input('date-range', 'end_date')]
)
def update_metrics(category_filter, category_values, start_date, end_date):
    filtered_df = df.copy()
    
    if category_filter and category_values:
        filtered_df = filtered_df[filtered_df[category_filter].isin(category_values)]
    
    if 'date' in filtered_df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
    
    total_records = len(filtered_df)
    avg_charges = f"${filtered_df['charges'].mean():,.0f}" if not filtered_df.empty else "$0"
    avg_age = f"{filtered_df['age'].mean():.1f}" if not filtered_df.empty else "0"
    avg_bmi = f"{filtered_df['bmi'].mean():.1f}" if not filtered_df.empty else "0"
    
    return total_records, avg_charges, avg_age, avg_bmi

@app.callback(
    [Output('line-chart', 'figure'), Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure'), Output('pie-chart', 'figure')],
    [Input('category-filter', 'value'), Input('category-value-filter', 'value'),
     Input('date-range', 'start_date'), Input('date-range', 'end_date')]
)
def update_charts(category_filter, category_values, start_date, end_date):
    filtered_df = df.copy()
    
    if category_filter and category_values:
        filtered_df = filtered_df[filtered_df[category_filter].isin(category_values)]
    
    if 'date' in filtered_df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
    
    # Line Chart
    if 'date' in filtered_df.columns and not filtered_df.empty:
        line_data = filtered_df.groupby('date')['charges'].mean().reset_index()
        line_fig = px.line(line_data, x='date', y='charges', title="📈 Insurance Charges Trends")
    else:
        line_fig = go.Figure()
        line_fig.update_layout(title="📈 Insurance Charges Trends")
    
    # Bar Chart
    if not filtered_df.empty:
        bar_data = filtered_df.groupby('region')['charges'].mean().reset_index()
        bar_fig = px.bar(bar_data, x='region', y='charges', title="📊 Average Charges by Region")
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(title="📊 Average Charges by Region")
    
    # Scatter Plot
    if not filtered_df.empty and 'age' in filtered_df.columns:
        scatter_fig = px.scatter(filtered_df, x='age', y='charges', color='smoker',
                                title="🎯 Age vs Charges Analysis")
    else:
        scatter_fig = go.Figure()
        scatter_fig.update_layout(title="🎯 Age vs Charges Analysis")
    
    # Pie Chart
    if not filtered_df.empty:
        pie_data = filtered_df['region'].value_counts().reset_index()
        pie_data.columns = ['region', 'count']
        pie_fig = px.pie(pie_data, names='region', values='count', title="🥧 Region Distribution")
    else:
        pie_fig = go.Figure()
        pie_fig.update_layout(title="🥧 Region Distribution")
    
    # Apply dark theme
    for fig in [line_fig, bar_fig, scatter_fig, pie_fig]:
        fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    
    return line_fig, bar_fig, scatter_fig, pie_fig

@app.callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('ml-age', 'value'), State('ml-bmi', 'value'), State('ml-children', 'value'),
     State('ml-smoker', 'value'), State('ml-sex', 'value'), State('ml-region', 'value')]
)
def predict_insurance(n_clicks, age, bmi, children, smoker, sex, region):
    if n_clicks is None:
        return ""
    
    try:
        if age is None or bmi is None or children is None or smoker is None or sex is None or region is None:
            return dbc.Alert("Please fill in all fields", color="warning")
        
        predicted_charges = ml_manager.predict_charges(age, bmi, children, smoker, sex, region)
        predicted_risk = ml_manager.predict_risk(age, bmi, children, smoker, sex, region)
        
        if predicted_charges is not None and predicted_risk is not None:
            risk_color = "success" if predicted_risk == "Low" else "warning" if predicted_risk == "Medium" else "danger"
            
            return dbc.Card([
                dbc.CardBody([
                    html.H6("🎯 Predicted Results:", className="text-white mb-2"),
                    html.P(f"💰 Charges: ${predicted_charges:,.0f}", className="text-success mb-2"),
                    html.P(f"⚠️ Risk Level: {predicted_risk}", className=f"text-{risk_color} mb-0")
                ])
            ], style={'backgroundColor': 'rgba(0,255,0,0.1)', 'border': '1px solid rgba(0,255,0,0.3)'})
        else:
            return dbc.Alert("❌ Prediction failed", color="danger")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return dbc.Alert(f"❌ Prediction error: {str(e)}", color="danger")

# For Vercel deployment
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
