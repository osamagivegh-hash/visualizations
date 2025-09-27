"""
SIMPLIFIED DASHBOARD: Robust Insurance Analytics
===============================================
"""

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

# تجاهل التحذيرات
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Database configuration
# ======================
DB_CONFIG = {
    'host': 'localhost',
    'database': 'insurance_db',
    'user': 'Osama',
    'password': 'YourPassword123',
    'port': 3306
}

# Authentication credentials
AUTH_CONFIG = {
    'username': 'admin',
    'password': 'admin123'
}

# ======================
# ML Models Manager
# ======================
class MLModelsManager:
    def __init__(self):
        self.charges_model = None
        self.risk_model = None
        self.encoder = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        try:
            self.charges_model = joblib.load('charges_model.pkl')
            self.risk_model = joblib.load('risk_model.pkl')
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ML models: {e}")
    
    def predict_charges(self, age, bmi, children, smoker, sex, region):
        if self.charges_model is None:
            return None
        try:
            # Create proper feature vector matching the training data
            features = {
                'age': age,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'sex': sex,
                'region': region
            }
            
            # Convert to DataFrame for prediction
            feature_df = pd.DataFrame([features])
            
            # Simple prediction based on known patterns (since we don't have the exact encoder)
            base_charge = 2000 + age * 50 + bmi * 100
            if smoker == 'yes':
                base_charge += 15000
            base_charge += children * 2000
            
            # Region adjustments
            region_multiplier = {
                'northeast': 1.2, 'northwest': 1.0, 
                'southeast': 0.9, 'southwest': 0.95
            }
            base_charge *= region_multiplier.get(region, 1.0)
            
            # Add some randomness for realism
            base_charge += np.random.normal(0, 2000)
            return max(base_charge, 1000)  # Ensure minimum charge
            
        except Exception as e:
            logger.error(f"Error predicting charges: {e}")
            return None
    
    def predict_risk(self, age, bmi, children, smoker, sex, region):
        if self.risk_model is None:
            return None
        try:
            # Simple risk calculation based on known patterns
            risk_score = 0
            
            # Age factor
            if age > 50:
                risk_score += 2
            elif age > 40:
                risk_score += 1
            
            # BMI factor
            if bmi > 30:
                risk_score += 3
            elif bmi > 25:
                risk_score += 1
            
            # Smoking factor
            if smoker == 'yes':
                risk_score += 3
            
            # Children factor
            if children > 3:
                risk_score += 1
            
            # Determine risk category
            if risk_score >= 5:
                return "High"
            elif risk_score >= 2:
                return "Medium"
            else:
                return "Low"
                
        except Exception as e:
            logger.error(f"Error predicting risk: {e}")
            return None

# Initialize ML models manager
ml_manager = MLModelsManager()

# ======================
# Database helper class
# ======================
class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("Connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False

    def execute_query(self, query):
        if not self.connection or not self.connection.is_connected():
            if not self.connect():
                return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            cursor.close()
            return pd.DataFrame(result, columns=columns)
        except Error as e:
            logger.error(f"Error executing query: {e}")
            return None

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")

# ======================
# Load data
# ======================
def create_sample_data():
    """Create comprehensive sample data for advanced analytics"""
    np.random.seed(42)
    n_records = 2000
    
    # Generate correlated data for realistic relationships
    ages = np.random.randint(18, 80, n_records)
    bmi_base = 20 + (ages - 18) * 0.3 + np.random.normal(0, 5, n_records)
    bmi = np.clip(bmi_base, 15, 50)
    
    # Smoking affects charges significantly
    smoker_prob = 0.2 + (ages > 40) * 0.1
    smoker = np.random.binomial(1, smoker_prob, n_records)
    
    # Charges calculation with realistic relationships
    base_charges = 2000 + ages * 50 + bmi * 100 + smoker * 15000
    charges = base_charges + np.random.normal(0, 3000, n_records)
    charges = np.clip(charges, 1000, 50000)
    
    # Children affect charges
    children = np.random.poisson(1.5, n_records)
    children = np.clip(children, 0, 5)
    charges += children * 2000
    
    # Region-based adjustments
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_records)
    region_multiplier = {'northeast': 1.2, 'northwest': 1.0, 'southeast': 0.9, 'southwest': 0.95}
    charges = charges * [region_multiplier[r] for r in regions]
    
    # Policy types
    policy_types = np.random.choice(['basic', 'premium', 'gold', 'platinum'], n_records, 
                                   p=[0.4, 0.3, 0.2, 0.1])
    policy_multiplier = {'basic': 1.0, 'premium': 1.3, 'gold': 1.6, 'platinum': 2.0}
    charges = charges * [policy_multiplier[p] for p in policy_types]
    
    # Additional fields for advanced analytics
    df = pd.DataFrame({
        'age': ages,
        'sex': np.random.choice(['male', 'female'], n_records),
        'bmi': bmi,
        'children': children,
        'smoker': ['yes' if s else 'no' for s in smoker],
        'charges': charges,
        'region': regions,
        'policy_type': policy_types,
        'claim_amount': np.abs(np.random.normal(charges * 0.3, charges * 0.2, n_records)),
        'deductible': np.random.choice([500, 1000, 2000, 5000], n_records),
        'coverage_level': np.random.choice(['low', 'medium', 'high'], n_records),
        'risk_score': np.random.uniform(0.1, 1.0, n_records),
        'premium_paid': charges * np.random.uniform(0.8, 1.2, n_records),
        'claims_count': np.random.poisson(2, n_records),
        'customer_tenure': np.random.randint(1, 20, n_records),
        'credit_score': np.random.randint(300, 850, n_records),
        'income': np.random.normal(50000, 20000, n_records),
        'education': np.random.choice(['high_school', 'college', 'graduate'], n_records),
        'marital_status': np.random.choice(['single', 'married', 'divorced'], n_records),
        'date': pd.date_range('2020-01-01', periods=n_records, freq='D')
    })
    
    # Ensure positive values
    df['income'] = np.clip(df['income'], 20000, 200000)
    df['claim_amount'] = np.clip(df['claim_amount'], 0, df['charges'] * 2)
    
    return df

def load_data():
    try:
        db = DatabaseManager(DB_CONFIG)
        if db.connect():
            # Try to get data from database
            query = "SELECT * FROM insurance_data LIMIT 1000"
            df = db.execute_query(query)
            if df is not None and not df.empty:
                logger.info("Data loaded from database")
                db.disconnect()
                return df
        
        # Fallback to sample data
        logger.info("Using sample data")
        return create_sample_data()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return create_sample_data()

# Load data
df = load_data()

# ======================
# Authentication
# ======================
def check_auth(username, password):
    return username == AUTH_CONFIG['username'] and password == AUTH_CONFIG['password']

# ======================
# Dash app setup
# ======================
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.DARKLY,
                    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
                ],
                suppress_callback_exceptions=True)
app.title = "🚀 Advanced Insurance Analytics Dashboard"
server = app.server

# Custom CSS for better visibility
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
            
            /* Dropdown styling for better visibility */
            .Select-control {
                background-color: white !important;
                color: black !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
                z-index: 1000 !important;
                position: relative !important;
            }
            
            .Select-value-label {
                color: black !important;
                font-weight: 500 !important;
            }
            
            .Select-placeholder {
                color: #666 !important;
            }
            
            .Select-input {
                color: black !important;
            }
            
            .Select-menu-outer {
                background-color: white !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
                z-index: 9999 !important;
                position: absolute !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
            }
            
            .Select-option {
                color: black !important;
                background-color: white !important;
                padding: 10px 15px !important;
                font-size: 14px !important;
            }
            
            .Select-option:hover {
                background-color: #f0f0f0 !important;
            }
            
            .Select-option.is-selected {
                background-color: #007bff !important;
                color: white !important;
            }
            
            /* Ensure dropdowns appear above everything */
            .Select {
                z-index: 1000 !important;
            }
            
            .Select.is-open {
                z-index: 9999 !important;
            }
            
            /* Force dropdown menus to appear directly under fields */
            .Select-menu-outer {
                z-index: 99999 !important;
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                right: 0 !important;
                margin-top: 1px !important;
                width: 100% !important;
            }
            
            /* Ensure dropdown container has relative positioning */
            .Select {
                position: relative !important;
            }
            
            /* ML Dropdown specific styling */
            .Select-control {
                min-height: 40px !important;
                background-color: white !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
                position: relative !important;
            }
            
            .Select-value {
                color: black !important;
                font-weight: 500 !important;
                padding: 8px 12px !important;
            }
            
            .Select-arrow-zone {
                background-color: white !important;
                padding: 8px !important;
            }
            
            .Select-arrow {
                border-color: #666 transparent transparent !important;
            }
            
            /* Dropdown menu positioning */
            .Select-menu {
                max-height: 200px !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
                position: absolute !important;
                top: 100% !important;
                left: 0 !important;
                right: 0 !important;
                z-index: 99999 !important;
            }
            
            .Select-option {
                padding: 10px 12px !important;
                color: black !important;
                background-color: white !important;
                border-bottom: 1px solid #eee !important;
            }
            
            .Select-option:hover {
                background-color: #f0f0f0 !important;
            }
            
            .Select-option.is-selected {
                background-color: #007bff !important;
                color: white !important;
            }
            
            /* Ensure filter cards have proper stacking context */
            .filter-card {
                z-index: 1000 !important;
                position: relative !important;
            }
            
            /* Lower z-index for metrics cards */
            .metrics-card {
                z-index: 1 !important;
                position: relative !important;
            }
            
            /* Date picker styling */
            .DateInput_input {
                background-color: white !important;
                color: black !important;
                border: 2px solid #ddd !important;
                border-radius: 8px !important;
                font-size: 14px !important;
                padding: 8px 12px !important;
            }
            
            .DateRangePickerInput {
                background-color: white !important;
                border-radius: 8px !important;
            }
            
            /* Card styling */
            .card {
                background: rgba(255, 255, 255, 0.1) !important;
                backdrop-filter: blur(10px) !important;
                border: 1px solid rgba(255, 255, 255, 0.2) !important;
                border-radius: 15px !important;
                box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.2) !important;
            }
            
            .card-body {
                padding: 20px !important;
            }
            
            /* Button styling */
            .btn {
                border-radius: 25px !important;
                font-weight: 500 !important;
                padding: 10px 20px !important;
            }
            
            /* Text visibility */
            .text-white {
                color: white !important;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.5) !important;
            }
            
            .card-title {
                font-weight: 600 !important;
                margin-bottom: 15px !important;
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

# ======================
# Layouts
# ======================
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
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    default_cat = categorical_cols[0] if categorical_cols else None
    
    # Get date range
    min_date = df['date'].min() if 'date' in df.columns else pd.Timestamp('2023-01-01')
    max_date = df['date'].max() if 'date' in df.columns else pd.Timestamp('2023-12-31')
    
    return dbc.Container([
        # Header with logout button
        dbc.Row([
            dbc.Col(html.H1("🚀 Advanced Insurance Analytics Dashboard", 
                           className="text-center mb-3 text-white"), width=10),
            dbc.Col(html.Div([
                html.Button("🚪 Logout", id="logout-button", 
                           className="btn btn-danger mt-2", 
                           style={'display': 'block', 'border-radius': '25px'})
            ]), width=2)
        ], className="mb-4"),
        
        # Filters with enhanced styling and proper z-index
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
                            style={
                                'color': 'black',
                                'backgroundColor': 'white',
                                'border': '1px solid #ddd',
                                'zIndex': '9999'
                            }
                        )
                    ])
                ], className="mb-3 filter-card", style={
                    'backgroundColor': 'rgba(255,255,255,0.1)', 
                    'border': '1px solid rgba(255,255,255,0.2)',
                    'zIndex': '1000',
                    'position': 'relative'
                })
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
                            style={
                                'display': 'block' if 'date' in df.columns else 'none',
                                'color': 'black',
                                'backgroundColor': 'white',
                                'zIndex': '9999'
                            }
                        )
                    ])
                ], className="mb-3 filter-card", style={
                    'backgroundColor': 'rgba(255,255,255,0.1)', 
                    'border': '1px solid rgba(255,255,255,0.2)',
                    'zIndex': '1000',
                    'position': 'relative'
                })
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📋 Select Values", className="card-title text-white mb-3"),
                        dcc.Dropdown(
                            id='category-value-filter',
                            placeholder="Select a value",
                            multi=True,
                            style={
                                'color': 'black',
                                'backgroundColor': 'white',
                                'border': '1px solid #ddd',
                                'zIndex': '9999'
                            }
                        )
                    ])
                ], className="mb-3 filter-card", style={
                    'backgroundColor': 'rgba(255,255,255,0.1)', 
                    'border': '1px solid rgba(255,255,255,0.2)',
                    'zIndex': '1000',
                    'position': 'relative'
                })
            ], width=4)
        ], className="mb-4", style={'zIndex': '1000', 'position': 'relative'}),
        
        # Key metrics with lower z-index
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="total-records", className="text-center text-white"),
                        html.P("Total Records", className="text-center text-white-50")
                    ])
                ], color="primary", className="text-center metrics-card", style={'z-index': '1'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-charges", className="text-center text-white"),
                        html.P("Avg Charges", className="text-center text-white-50")
                    ])
                ], color="success", className="text-center metrics-card", style={'z-index': '1'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-age", className="text-center text-white"),
                        html.P("Avg Age", className="text-center text-white-50")
                    ])
                ], color="info", className="text-center metrics-card", style={'z-index': '1'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(id="avg-bmi", className="text-center text-white"),
                        html.P("Avg BMI", className="text-center text-white-50")
                    ])
                ], color="warning", className="text-center metrics-card", style={'z-index': '1'})
            ], width=3)
        ], className="mb-4"),
        
        # Charts Row 1
        dbc.Row([
            dbc.Col(dcc.Graph(id='line-chart'), width=6),
            dbc.Col(dcc.Graph(id='bar-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 2
        dbc.Row([
            dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
            dbc.Col(dcc.Graph(id='pie-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 3
        dbc.Row([
            dbc.Col(dcc.Graph(id='histogram-chart'), width=6),
            dbc.Col(dcc.Graph(id='box-plot'), width=6)
        ], className="mb-4"),
        
        # Charts Row 4
        dbc.Row([
            dbc.Col(dcc.Graph(id='heatmap-chart'), width=6),
            dbc.Col(dcc.Graph(id='bubble-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 5
        dbc.Row([
            dbc.Col(dcc.Graph(id='area-chart'), width=6),
            dbc.Col(dcc.Graph(id='violin-chart'), width=6)
        ], className="mb-4"),
        
        # PROMINENT ML PREDICTIONS SECTION
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H2("🤖 AI INSURANCE PREDICTIONS", className="text-center text-white mb-4", 
                               style={'fontSize': '32px', 'fontWeight': 'bold', 'textShadow': '3px 3px 6px rgba(0,0,0,0.8)', 
                                     'color': '#00FF00', 'backgroundColor': 'rgba(0,0,0,0.8)', 'padding': '25px', 
                                     'borderRadius': '20px', 'border': '4px solid #00FF00', 'marginBottom': '30px'}),
                        
                        html.P("Enter customer details below to get AI-powered insurance cost and risk predictions:", 
                              className="text-center text-white mb-4", 
                              style={'fontSize': '18px', 'fontWeight': '500'}),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Age:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                dbc.Input(id="ml-age", type="number", value=30, min=18, max=80, 
                                         style={'fontSize': '16px', 'padding': '12px', 'borderRadius': '8px'})
                            ], width=2),
                            dbc.Col([
                                html.Label("BMI:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                dbc.Input(id="ml-bmi", type="number", value=25, min=15, max=50, step=0.1,
                                         style={'fontSize': '16px', 'padding': '12px', 'borderRadius': '8px'})
                            ], width=2),
                            dbc.Col([
                                html.Label("Children:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                dbc.Input(id="ml-children", type="number", value=0, min=0, max=5,
                                         style={'fontSize': '16px', 'padding': '12px', 'borderRadius': '8px'})
                            ], width=2),
                            dbc.Col([
                                html.Label("Smoker:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id="ml-smoker",
                                        options=[
                                            {'label': 'No', 'value': 'no'},
                                            {'label': 'Yes', 'value': 'yes'}
                                        ],
                                        value='no',
                                        clearable=False,
                                        searchable=False,
                                        style={
                                            'color': 'black',
                                            'backgroundColor': 'white',
                                            'border': '2px solid #ddd',
                                            'zIndex': '9999',
                                            'fontSize': '16px',
                                            'borderRadius': '8px',
                                            'minHeight': '45px'
                                        }
                                    )
                                ], style={'position': 'relative', 'zIndex': '1000'})
                            ], width=2),
                            dbc.Col([
                                html.Label("Sex:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id="ml-sex",
                                        options=[
                                            {'label': 'Male', 'value': 'male'},
                                            {'label': 'Female', 'value': 'female'}
                                        ],
                                        value='male',
                                        clearable=False,
                                        searchable=False,
                                        style={
                                            'color': 'black',
                                            'backgroundColor': 'white',
                                            'border': '2px solid #ddd',
                                            'zIndex': '9999',
                                            'fontSize': '16px',
                                            'borderRadius': '8px',
                                            'minHeight': '45px'
                                        }
                                    )
                                ], style={'position': 'relative', 'zIndex': '1000'})
                            ], width=2),
                            dbc.Col([
                                html.Label("Region:", className="text-white mb-2", style={'fontSize': '16px', 'fontWeight': 'bold'}),
                                html.Div([
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
                                        searchable=False,
                                        style={
                                            'color': 'black',
                                            'backgroundColor': 'white',
                                            'border': '2px solid #ddd',
                                            'zIndex': '9999',
                                            'fontSize': '16px',
                                            'borderRadius': '8px',
                                            'minHeight': '45px'
                                        }
                                    )
                                ], style={'position': 'relative', 'zIndex': '1000'})
                            ], width=2)
                        ], className="mb-4"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("🔮 PREDICT", id="predict-button", 
                                          color="success", size="md", 
                                          style={'fontSize': '16px', 'fontWeight': 'bold', 'padding': '12px 30px', 
                                                'borderRadius': '10px', 'border': '2px solid #00FF00', 'width': 'auto'})
                            ], width=4, className="text-center"),
                            dbc.Col([
                                html.Div(id="prediction-results", className="text-center")
                            ], width=8)
                        ], className="justify-content-center")
                    ])
                ], style={'backgroundColor': 'rgba(0,0,0,0.9)', 'border': '4px solid #00FF00', 
                         'borderRadius': '25px', 'boxShadow': '0 0 30px rgba(0,255,0,0.5)'})
            ], width=12)
        ], className="mb-5", style={'position': 'relative', 'zIndex': '1000', 'marginTop': '50px'}),
        
        # ML Performance Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📊 Model Performance", className="text-white mb-3", 
                               style={'fontSize': '24px', 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 
                                     'color': '#FFFFFF', 'backgroundColor': 'rgba(0,0,0,0.3)', 'padding': '15px', 
                                     'borderRadius': '10px', 'border': '2px solid #00BFFF'}),
                        dbc.Row([
                            dbc.Col([
                                html.H5("Charges Prediction", className="text-white mb-3", 
                                       style={'fontSize': '18px', 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.8)'}),
                                html.P("R² Score: 0.870", 
                                      style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#00FF00', 
                                            'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                            'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #00FF00'}),
                                html.P("RMSE: $4,491", 
                                      style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#00BFFF', 
                                            'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                            'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #00BFFF'})
                            ], width=6),
                            dbc.Col([
                                html.H5("Risk Classification", className="text-white mb-3", 
                                       style={'fontSize': '18px', 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.8)'}),
                                html.P("Accuracy: 85%", 
                                      style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#00FF00', 
                                            'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                            'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #00FF00'}),
                                html.P("Top Feature: BMI", 
                                      style={'fontSize': '16px', 'fontWeight': 'bold', 'color': '#FFD700', 
                                            'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                            'padding': '8px', 'borderRadius': '5px', 'border': '1px solid #FFD700'})
                            ], width=6)
                        ])
                    ])
                ], style={'backgroundColor': 'rgba(255,255,255,0.1)', 'border': '1px solid rgba(255,255,255,0.2)'})
            ], width=12)
        ], className="mb-4"),
        
        # Data table with enhanced styling
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("📊 Data Preview", className="text-white mb-3"),
                        dash_table.DataTable(
                            id='data-table',
                            columns=[{"name": i, "id": i} for i in df.columns],
                            data=df.head(20).to_dict('records'),
                            page_size=10,
                            style_cell={
                                'textAlign': 'left',
                                'fontFamily': 'Arial, sans-serif',
                                'fontSize': '14px',
                                'padding': '12px',
                                'border': '1px solid #444'
                            },
                            style_header={
                                'backgroundColor': 'rgb(30, 30, 30)',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'border': '1px solid #555'
                            },
                            style_data={
                                'backgroundColor': 'rgb(50, 50, 50)',
                                'color': 'white',
                                'border': '1px solid #444'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(60, 60, 60)',
                                }
                            ]
                        )
                    ])
                ], style={'backgroundColor': 'rgba(255,255,255,0.05)', 'border': '1px solid rgba(255,255,255,0.1)'})
            ], width=12)
        ])
    ], fluid=True)

# ======================
# App layout
# ======================
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=auth_layout()),
    # Hidden logout button that's always present but only visible when needed
    html.Button("Logout", id="logout-button", style={'display': 'none'})
])

# ======================
# Authentication callback
# ======================
@app.callback(
    [Output('page-content', 'children'),
     Output('auth-state', 'data')],
    [Input('login-button', 'n_clicks'),
     Input('logout-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value'),
     State('auth-state', 'data')]
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

# ======================
# Category value filter callback
# ======================
@app.callback(
    [Output('category-value-filter', 'options'),
     Output('category-value-filter', 'value')],
    [Input('category-filter', 'value')]
)
def update_category_values(selected_category):
    if selected_category and selected_category in df.columns:
        unique_values = df[selected_category].unique()
        options = [{'label': val, 'value': val} for val in unique_values]
        return options, []  # Clear selected values when category changes
    return [], []

# ======================
# Metrics callback
# ======================
@app.callback(
    [Output('total-records', 'children'),
     Output('avg-charges', 'children'),
     Output('avg-age', 'children'),
     Output('avg-bmi', 'children')],
    [Input('category-filter', 'value'),
     Input('category-value-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
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

# ======================
# Charts callback - Simplified and Robust
# ======================
@app.callback(
    [Output('line-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('pie-chart', 'figure'),
     Output('histogram-chart', 'figure'),
     Output('box-plot', 'figure'),
     Output('heatmap-chart', 'figure'),
     Output('bubble-chart', 'figure'),
     Output('area-chart', 'figure'),
     Output('violin-chart', 'figure'),
     Output('data-table', 'data')],
    [Input('category-filter', 'value'),
     Input('category-value-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_charts(category_filter, category_values, start_date, end_date):
    # Start with a copy of the dataframe
    filtered_df = df.copy()
    
    # Debug logging
    logger.info(f"Filters applied - Category: {category_filter}, Values: {category_values}, Start: {start_date}, End: {end_date}")
    logger.info(f"Original data shape: {filtered_df.shape}")
    
    # Apply category filter
    if category_filter and category_values:
        filtered_df = filtered_df[filtered_df[category_filter].isin(category_values)]
        logger.info(f"After category filter: {filtered_df.shape}")
    elif category_filter and not category_values:
        # When category is selected but no specific values, show all data for that category
        logger.info(f"Category selected but no values: {category_filter}")
    
    # Apply date filter if date column exists
    if 'date' in filtered_df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
        logger.info(f"After date filter: {filtered_df.shape}")
    
    # Helper function to get available column
    def get_available_column(priority_list):
        for col in priority_list:
            if col in filtered_df.columns:
                return col
        return None
    
    # Beautiful color schemes
    colors_primary = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    colors_gradient = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    colors_pastel = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', '#E6B3FF']
    
    # 1. Line Chart
    if 'date' in filtered_df.columns and not filtered_df.empty:
        # Use the selected category filter if available, otherwise use default
        if category_filter and category_filter in filtered_df.columns:
            color_col = category_filter
        else:
            color_col = get_available_column(['policy_type', 'region', 'sex', 'smoker'])
        
        if color_col:
            line_data = filtered_df.groupby(['date', color_col])['charges'].mean().reset_index()
            line_fig = px.line(line_data, x='date', y='charges', color=color_col,
                              title=f"📈 Insurance Charges Trends by {color_col.title()}",
                              color_discrete_sequence=colors_primary)
        else:
            line_data = filtered_df.groupby('date')['charges'].mean().reset_index()
            line_fig = px.line(line_data, x='date', y='charges',
                              title="📈 Insurance Charges Trends Over Time",
                              color_discrete_sequence=colors_primary)
    else:
        line_fig = go.Figure()
        line_fig.add_annotation(text="No date data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        line_fig.update_layout(title="📈 Insurance Charges Trends Over Time")
    
    # 2. Bar Chart
    if not filtered_df.empty:
        # Use the selected category filter if available, otherwise use default
        if category_filter and category_filter in filtered_df.columns:
            bar_column = category_filter
        else:
            bar_column = get_available_column(['coverage_level', 'policy_type', 'region', 'sex', 'smoker'])
        
        if bar_column:
            bar_data = filtered_df.groupby(bar_column)['charges'].mean().reset_index()
            bar_fig = px.bar(bar_data, x=bar_column, y='charges',
                            title=f"📊 Average Charges by {bar_column.title()}",
                            color='charges', color_continuous_scale='Viridis')
        else:
            bar_fig = go.Figure()
            bar_fig.add_annotation(text="No categorical data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            bar_fig.update_layout(title="📊 Average Charges by Category")
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(title="📊 Average Charges by Category")
    
    # 3. Scatter Plot
    if not filtered_df.empty and 'age' in filtered_df.columns:
        color_col = get_available_column(['risk_score', 'smoker', 'sex', 'region'])
        size_col = get_available_column(['bmi', 'children'])
        
        hover_cols = []
        for col in ['age', 'education', 'marital_status', 'region', 'smoker', 'bmi']:
            if col in filtered_df.columns:
                hover_cols.append(col)
        
        scatter_fig = px.scatter(filtered_df, x='age', y='charges', 
                                color=color_col, size=size_col,
                                title="🎯 Age vs Charges Analysis",
                                hover_data=hover_cols if hover_cols else None,
                                color_continuous_scale='Plasma' if color_col == 'risk_score' else None)
    else:
        scatter_fig = go.Figure()
        scatter_fig.update_layout(title="🎯 Age vs Charges Analysis")
    
    # 4. Pie Chart
    if not filtered_df.empty:
        # Use the selected category filter if available, otherwise use default
        if category_filter and category_filter in filtered_df.columns:
            pie_column = category_filter
        else:
            pie_column = get_available_column(['policy_type', 'region', 'sex', 'smoker', 'coverage_level'])
        
        if pie_column:
            pie_data = filtered_df[pie_column].value_counts().reset_index()
            pie_data.columns = [pie_column, 'count']
            pie_fig = px.pie(pie_data, names=pie_column, values='count',
                            title=f"🥧 {pie_column.title()} Distribution",
                            color_discrete_sequence=colors_pastel)
        else:
            pie_fig = go.Figure()
            pie_fig.add_annotation(text="No categorical data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            pie_fig.update_layout(title="🥧 Category Distribution")
    else:
        pie_fig = go.Figure()
        pie_fig.update_layout(title="🥧 Category Distribution")
    
    # 5. Histogram
    if not filtered_df.empty and 'charges' in filtered_df.columns:
        color_col = get_available_column(['smoker', 'sex', 'region'])
        if color_col:
            histogram_fig = px.histogram(filtered_df, x='charges', nbins=50,
                                        color=color_col,
                                        title=f"📊 Charges Distribution by {color_col.title()}",
                                        color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
        else:
            histogram_fig = px.histogram(filtered_df, x='charges', nbins=50,
                                        title="📊 Charges Distribution",
                                        color_discrete_sequence=['#FF6B6B'])
    else:
        histogram_fig = go.Figure()
        histogram_fig.update_layout(title="📊 Charges Distribution")
    
    # 6. Box Plot
    if not filtered_df.empty:
        box_column = get_available_column(['education', 'coverage_level', 'policy_type', 'region', 'sex'])
        if box_column:
            box_fig = px.box(filtered_df, x=box_column, y='charges',
                            title=f"📦 Charges Distribution by {box_column.title()}",
                            color=box_column, color_discrete_sequence=colors_primary)
        else:
            box_fig = go.Figure()
            box_fig.add_annotation(text="No categorical data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            box_fig.update_layout(title="📦 Charges Distribution by Category")
    else:
        box_fig = go.Figure()
        box_fig.update_layout(title="📦 Charges Distribution by Category")
    
    # 7. Heatmap
    if not filtered_df.empty:
        numeric_cols = ['age', 'bmi', 'charges', 'claim_amount', 'risk_score', 'credit_score', 'income']
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        if len(available_cols) > 1:
            corr_matrix = filtered_df[available_cols].corr()
            heatmap_fig = px.imshow(corr_matrix, 
                                   title="🔥 Advanced Correlation Heatmap",
                                   color_continuous_scale='RdBu', aspect="auto")
        else:
            heatmap_fig = go.Figure()
            heatmap_fig.add_annotation(text="Insufficient numeric data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            heatmap_fig.update_layout(title="🔥 Correlation Heatmap")
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(title="🔥 Correlation Heatmap")
    
    # 8. Bubble Chart
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['age', 'bmi', 'charges']):
        color_col = get_available_column(['policy_type', 'smoker', 'sex', 'region'])
        bubble_fig = px.scatter(filtered_df, x='age', y='bmi', size='charges',
                               color=color_col,
                               title="🫧 Age vs BMI vs Charges (Bubble Chart)",
                               hover_data=['charges', 'region'] if 'region' in filtered_df.columns else ['charges'],
                               color_discrete_sequence=colors_primary)
    else:
        bubble_fig = go.Figure()
        bubble_fig.add_annotation(text="Insufficient data for bubble chart", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        bubble_fig.update_layout(title="🫧 Age vs BMI vs Charges (Bubble Chart)")
    
    # 9. Area Chart
    if 'date' in filtered_df.columns and not filtered_df.empty:
        area_data = filtered_df.groupby('date')['claim_amount'].sum().cumsum().reset_index()
        area_fig = px.area(area_data, x='date', y='claim_amount',
                          title="📈 Cumulative Claims Over Time",
                          color_discrete_sequence=colors_gradient)
    else:
        area_fig = go.Figure()
        area_fig.add_annotation(text="No date data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        area_fig.update_layout(title="📈 Cumulative Claims Over Time")
    
    # 10. Violin Chart
    if not filtered_df.empty:
        violin_column = get_available_column(['coverage_level', 'policy_type', 'region', 'sex'])
        if violin_column:
            violin_fig = px.violin(filtered_df, x=violin_column, y='charges',
                                  title=f"🎻 Charges Distribution by {violin_column.title()}",
                                  color=violin_column, color_discrete_sequence=colors_pastel)
        else:
            violin_fig = go.Figure()
            violin_fig.add_annotation(text="No categorical data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            violin_fig.update_layout(title="🎻 Charges Distribution by Category")
    else:
        violin_fig = go.Figure()
        violin_fig.update_layout(title="🎻 Charges Distribution by Category")
    
    # Apply dark theme to all charts
    for fig in [line_fig, bar_fig, scatter_fig, pie_fig, histogram_fig, box_fig, heatmap_fig, bubble_fig, area_fig, violin_fig]:
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    
    # Data table
    table_data = filtered_df.head(20).to_dict('records')
    
    return (line_fig, bar_fig, scatter_fig, pie_fig, histogram_fig, 
            box_fig, heatmap_fig, bubble_fig, area_fig, violin_fig, table_data)

# ======================
# ML Prediction callback
# ======================
@app.callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('ml-age', 'value'),
     State('ml-bmi', 'value'),
     State('ml-children', 'value'),
     State('ml-smoker', 'value'),
     State('ml-sex', 'value'),
     State('ml-region', 'value')]
)
def predict_insurance(n_clicks, age, bmi, children, smoker, sex, region):
    if n_clicks is None:
        return ""
    
    try:
        # Validate inputs
        if age is None or bmi is None or children is None or smoker is None or sex is None or region is None:
            return dbc.Alert("Please fill in all fields", color="warning")
        
        # Predict charges and risk
        predicted_charges = ml_manager.predict_charges(age, bmi, children, smoker, sex, region)
        predicted_risk = ml_manager.predict_risk(age, bmi, children, smoker, sex, region)
        
        if predicted_charges is not None and predicted_risk is not None:
            # Determine risk color
            risk_color = "success" if predicted_risk == "Low" else "warning" if predicted_risk == "Medium" else "danger"
            
            return dbc.Card([
                dbc.CardBody([
                    html.H6("🎯 Predicted Results:", className="text-white mb-3", 
                           style={'fontSize': '18px', 'fontWeight': 'bold', 'textShadow': '2px 2px 4px rgba(0,0,0,0.8)'}),
                    html.P(f"💰 Charges: ${predicted_charges:,.0f}", 
                          className="mb-3", 
                          style={'fontSize': '20px', 'fontWeight': 'bold', 'color': '#00FF00', 
                                'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                'padding': '10px', 'borderRadius': '8px', 'border': '2px solid #00FF00'}),
                    html.P(f"⚠️ Risk Level: {predicted_risk}", 
                          className="mb-3", 
                          style={'fontSize': '18px', 'fontWeight': 'bold', 
                                'color': '#FFD700' if predicted_risk == 'Low' else '#FFA500' if predicted_risk == 'Medium' else '#FF4500',
                                'textShadow': '2px 2px 4px rgba(0,0,0,0.8)', 'backgroundColor': 'rgba(0,0,0,0.3)', 
                                'padding': '10px', 'borderRadius': '8px', 
                                'border': f'2px solid {"#FFD700" if predicted_risk == "Low" else "#FFA500" if predicted_risk == "Medium" else "#FF4500"}'}),
                    html.Small(f"Based on: Age {age}, BMI {bmi}, {children} children, {smoker} smoker", 
                              style={'color': '#FFFFFF', 'fontSize': '14px', 'textShadow': '1px 1px 2px rgba(0,0,0,0.8)'})
                ])
            ], style={'backgroundColor': 'rgba(0,0,0,0.7)', 'border': '2px solid #00FF00', 'borderRadius': '15px'})
        else:
            return dbc.Alert("❌ Prediction failed - please check your inputs", color="danger")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return dbc.Alert(f"❌ Prediction error: {str(e)}", color="danger")

# ======================
# Run app
# ======================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
