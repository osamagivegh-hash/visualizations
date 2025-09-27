"""
FINAL DASHBOARD: Secure & Interactive Insurance Analytics
=========================================================
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
# Database helper class
# ======================
class DatabaseManager:
    def __init__(self, config):
        self.config = config
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            return self.connection.is_connected()
        except Error as e:
            logger.error(f"MySQL Connection Error: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def execute_query(self, query):
        if not self.connect():
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Query Error: {e}")
            return None
        finally:
            self.disconnect()

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
        
        # Check if database exists and has tables
        tables = db.execute_query("SHOW TABLES")
        if not tables:
            logger.warning("No tables found. Using sample data.")
            return create_sample_data()
        
        table_name = list(tables[0].values())[0] if tables else None
        if not table_name:
            return create_sample_data()
            
        data = db.execute_query(f"SELECT * FROM {table_name}")
        if not data:
            return create_sample_data()
            
        df = pd.DataFrame(data)
        
        if df.empty:
            return create_sample_data()
            
        # Convert datetime columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return create_sample_data()

df = load_data()

# ======================
# Authentication check
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

# Custom CSS for enhanced styling
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
            .main-container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                margin: 20px;
                padding: 20px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                border: 1px solid rgba(255, 255, 255, 0.18);
            }
            .chart-container {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.2);
            }
            .metric-card {
                background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.3);
                margin: 10px 0;
            }
            .filter-card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .navbar {
                background: linear-gradient(90deg, #667eea, #764ba2) !important;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);
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
    alert = dbc.Alert(message, color="danger", dismissable=True) if message else ""
    return dbc.Container([
        html.Div([
            html.H1("Insurance Analytics Dashboard", className="text-center mb-4"),
            alert,
            dbc.Card([
                dbc.CardBody([
                    html.H4("Login", className="text-center mb-3"),
                    dbc.Input(id="username-input", placeholder="Username", type="text", className="mb-3"),
                    dbc.Input(id="password-input", placeholder="Password", type="password", className="mb-3"),
                    dbc.Button("Login", id="login-button", color="primary", className="w-100")
                ])
            ], style={"maxWidth": "400px", "margin": "0 auto"})
        ], style={"marginTop": "100px"})
    ], fluid=True)

def dashboard_layout():
    # Get categorical columns for filters
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
        
        # Filters
        dbc.Row([
            dbc.Col([
                html.Label("Filter by Category:"),
                dcc.Dropdown(
                    id='category-filter',
                    options=[{'label': col, 'value': col} for col in categorical_cols],
                    placeholder="Select a category",
                    value=default_cat
                )
            ], width=4),
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=min_date,
                    end_date=max_date,
                    display_format='YYYY-MM-DD',
                    style={'display': 'block' if 'date' in df.columns else 'none'}
                )
            ], width=4),
            dbc.Col([
                html.Label("Category Value:"),
                dcc.Dropdown(
                    id='category-value-filter',
                    placeholder="Select a value",
                    multi=True
                )
            ], width=4) if categorical_cols else dbc.Col(width=4)
        ], className="mb-4"),
        
        # Key Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Total Records", className="card-title"),
                        html.H2(id="total-records", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Average Charges", className="card-title"),
                        html.H2(id="avg-charges", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Average Age", className="card-title"),
                        html.H2(id="avg-age", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Average BMI", className="card-title"),
                        html.H2(id="avg-bmi", className="card-text")
                    ])
                ])
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
        
        # Charts Row 6
        dbc.Row([
            dbc.Col(dcc.Graph(id='sunburst-chart'), width=6),
            dbc.Col(dcc.Graph(id='treemap-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 7
        dbc.Row([
            dbc.Col(dcc.Graph(id='radar-chart'), width=6),
            dbc.Col(dcc.Graph(id='polar-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 8
        dbc.Row([
            dbc.Col(dcc.Graph(id='waterfall-chart'), width=6),
            dbc.Col(dcc.Graph(id='funnel-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 9
        dbc.Row([
            dbc.Col(dcc.Graph(id='sankey-chart'), width=6),
            dbc.Col(dcc.Graph(id='candlestick-chart'), width=6)
        ], className="mb-4"),
        
        # Charts Row 10
        dbc.Row([
            dbc.Col(dcc.Graph(id='contour-chart'), width=6),
            dbc.Col(dcc.Graph(id='surface-chart'), width=6)
        ], className="mb-4"),
        
        # Data table
        dbc.Row([
            dbc.Col([
                html.H4("Data Preview"),
                dash_table.DataTable(
                    id='data-table',
                    page_size=10,
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    style_data={
                        'border': '1px solid lightgrey'
                    }
                )
            ])
        ])
    ], fluid=True)

# ======================
# App main layout
# ======================
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', children=auth_layout()),  # وضع layout المصادقة كقيمة ابتدائية
    # Hidden logout button that's always present but only visible when needed
    html.Button("Logout", id="logout-button", style={'display': 'none'})
])

# ======================
# Combined Authentication callback - حل نهائي
# ======================
@app.callback(
    Output('page-content', 'children'),
    Output('auth-state', 'data'),
    [Input('login-button', 'n_clicks'),
     Input('logout-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value'),
     State('auth-state', 'data')]
)
def handle_authentication(login_clicks, logout_clicks, username, password, auth_data):
    ctx = callback_context
    
    # إذا لم يتم تفعيل أي callback، ابق على الوضع الحالي
    if not ctx.triggered:
        return auth_layout(), auth_data
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # إذا ضغط على زر تسجيل الخروج
    if triggered_id == 'logout-button' and logout_clicks:
        return auth_layout(), {'authenticated': False}
    
    # إذا ضغط على زر الدخول
    elif triggered_id == 'login-button' and login_clicks:
        if check_auth(username, password):
            return dashboard_layout(), {'authenticated': True}
        else:
            return auth_layout("Invalid username or password"), {'authenticated': False}
    
    # إذا كان المستخدم مسجلاً بالفعل، اعرض الداشبورد
    if auth_data.get('authenticated', False):
        return dashboard_layout(), auth_data
    
    # الافتراضي: عرض صفحة الدخول
    return auth_layout(), auth_data

# ======================
# Update category values based on selected category
# ======================
@app.callback(
    Output('category-value-filter', 'options'),
    Input('category-filter', 'value')
)
def update_category_values(selected_category):
    if selected_category and selected_category in df.columns:
        values = df[selected_category].unique()
        return [{'label': str(val), 'value': str(val)} for val in values]
    return []

# ======================
# Update metrics callback
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
    
    # Apply category filter
    if category_filter and category_values:
        filtered_df = filtered_df[filtered_df[category_filter].isin(category_values)]
    
    # Apply date filter if date column exists
    if 'date' in filtered_df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
    
    total_records = len(filtered_df)
    avg_charges = f"${filtered_df['charges'].mean():,.2f}" if not filtered_df.empty else "$0"
    avg_age = f"{filtered_df['age'].mean():.1f}" if not filtered_df.empty else "0"
    avg_bmi = f"{filtered_df['bmi'].mean():.1f}" if not filtered_df.empty else "0"
    
    return total_records, avg_charges, avg_age, avg_bmi

# ======================
# Advanced Analytics Dashboard with 20 Visualizations
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
     Output('sunburst-chart', 'figure'),
     Output('treemap-chart', 'figure'),
     Output('radar-chart', 'figure'),
     Output('polar-chart', 'figure'),
     Output('waterfall-chart', 'figure'),
     Output('funnel-chart', 'figure'),
     Output('sankey-chart', 'figure'),
     Output('candlestick-chart', 'figure'),
     Output('contour-chart', 'figure'),
     Output('surface-chart', 'figure'),
     Output('data-table', 'data')],
    [Input('category-filter', 'value'),
     Input('category-value-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_charts(category_filter, category_values, start_date, end_date):
    # Start with a copy of the dataframe
    filtered_df = df.copy()
    
    # Apply category filter
    if category_filter and category_values:
        filtered_df = filtered_df[filtered_df[category_filter].isin(category_values)]
    
    # Apply date filter if date column exists
    if 'date' in filtered_df.columns and start_date and end_date:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
    
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
    
    # 1. Line Chart - Multi-dimensional charges analysis
    if 'date' in filtered_df.columns and not filtered_df.empty:
        # Use available categorical column for color
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
        line_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        line_fig = go.Figure()
        line_fig.add_annotation(text="No date data available", 
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
        line_fig.update_layout(title="📈 Insurance Charges Trends Over Time")
    
    # 2. Bar Chart - Risk vs Charges analysis
    if not filtered_df.empty:
        # Use available categorical column for bar chart
        bar_column = get_available_column(['coverage_level', 'policy_type', 'region', 'sex', 'smoker'])
        if bar_column:
            risk_data = filtered_df.groupby(bar_column)['charges'].mean().reset_index()
            bar_fig = px.bar(risk_data, x=bar_column, y='charges',
                            title=f"📊 Average Charges by {bar_column.title()}",
                            color='charges', 
                            color_continuous_scale='Viridis')
            bar_fig.update_layout(
                title_x=0.5, 
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        else:
            bar_fig = go.Figure()
            bar_fig.add_annotation(text="No categorical data available", 
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
            bar_fig.update_layout(title="📊 Average Charges by Category")
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(title="📊 Average Charges by Category")
    
    # 3. Scatter Plot - Age vs Charges with available variables
    if not filtered_df.empty and 'age' in filtered_df.columns:
        # Determine available columns for color and size
        color_col = get_available_column(['risk_score', 'smoker', 'sex', 'region'])
        size_col = get_available_column(['bmi', 'children'])
        
        # Determine hover data
        hover_cols = []
        for col in ['age', 'education', 'marital_status', 'region', 'smoker', 'bmi']:
            if col in filtered_df.columns:
                hover_cols.append(col)
        
        scatter_fig = px.scatter(filtered_df, x='age', y='charges', 
                                color=color_col, size=size_col,
                                title="🎯 Age vs Charges Analysis",
                                hover_data=hover_cols if hover_cols else None,
                                color_continuous_scale='Plasma' if color_col == 'risk_score' else None)
        scatter_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        scatter_fig = go.Figure()
        scatter_fig.update_layout(title="🎯 Age vs Charges Analysis")
    
    # 4. Pie Chart - Available categorical distribution
    if not filtered_df.empty:
        # Use available categorical column
        pie_column = get_available_column(['policy_type', 'region', 'sex', 'smoker', 'coverage_level'])
        if pie_column:
            pie_data = filtered_df[pie_column].value_counts().reset_index()
            pie_data.columns = [pie_column, 'count']
            pie_fig = px.pie(pie_data, names=pie_column, values='count',
                            title=f"🥧 {pie_column.title()} Distribution",
                            color_discrete_sequence=colors_pastel)
            pie_fig.update_layout(
                title_x=0.5, 
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        else:
            pie_fig = go.Figure()
            pie_fig.add_annotation(text="No categorical data available", 
                                  xref="paper", yref="paper",
                                  x=0.5, y=0.5, showarrow=False)
            pie_fig.update_layout(title="🥧 Category Distribution")
    else:
        pie_fig = go.Figure()
        pie_fig.update_layout(title="🥧 Category Distribution")
    
    # 5. Histogram - Charges distribution with multiple variables
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
        histogram_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        histogram_fig = go.Figure()
        histogram_fig.update_layout(title="📊 Charges Distribution")
    
    # 6. Box Plot - Charges by education level
    if not filtered_df.empty and 'education' in filtered_df.columns:
        box_fig = px.box(filtered_df, x='education', y='charges',
                        title="📦 Charges Distribution by Education Level",
                        color='education',
                        color_discrete_sequence=colors_primary)
        box_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        box_fig = go.Figure()
        box_fig.update_layout(title="📦 Charges Distribution by Education")
    
    # 7. Heatmap - Advanced correlation matrix
    if not filtered_df.empty:
        numeric_cols = ['age', 'bmi', 'charges', 'claim_amount', 'risk_score', 'credit_score', 'income']
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        if len(available_cols) > 1:
            corr_matrix = filtered_df[available_cols].corr()
            heatmap_fig = px.imshow(corr_matrix, 
                                   title="🔥 Advanced Correlation Heatmap",
                                   color_continuous_scale='RdBu',
                                   aspect="auto")
            heatmap_fig.update_layout(
                title_x=0.5, 
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
        else:
            heatmap_fig = go.Figure()
            heatmap_fig.update_layout(title="🔥 Correlation Heatmap")
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(title="🔥 Correlation Heatmap")
    
    # 8. Bubble Chart - Age vs Credit Score vs Charges
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['age', 'credit_score', 'charges']):
        bubble_fig = px.scatter(filtered_df, x='age', y='credit_score', size='charges',
                               color='policy_type',
                               title="🫧 Age vs Credit Score vs Charges",
                               hover_data=['income', 'education', 'marital_status'],
                               color_discrete_sequence=colors_primary)
        bubble_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        bubble_fig = go.Figure()
        bubble_fig.update_layout(title="🫧 Age vs Credit Score vs Charges")
    
    # 9. Area Chart - Cumulative claims over time
    if 'date' in filtered_df.columns and not filtered_df.empty:
        area_data = filtered_df.groupby('date')['claim_amount'].sum().cumsum().reset_index()
        area_fig = px.area(area_data, x='date', y='claim_amount',
                          title="📈 Cumulative Claims Over Time",
                          color_discrete_sequence=colors_gradient)
        area_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        area_fig = go.Figure()
        area_fig.update_layout(title="📈 Cumulative Claims Over Time")
    
    # 10. Violin Chart - Risk score distribution
    if not filtered_df.empty and 'risk_score' in filtered_df.columns:
        violin_fig = px.violin(filtered_df, x='coverage_level', y='risk_score',
                              title="🎻 Risk Score Distribution by Coverage Level",
                              color='coverage_level',
                              color_discrete_sequence=colors_pastel)
        violin_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        violin_fig = go.Figure()
        violin_fig.update_layout(title="🎻 Risk Score Distribution")
    
    # 11. Sunburst Chart - Hierarchical analysis
    if not filtered_df.empty:
        sunburst_data = filtered_df.groupby(['region', 'policy_type', 'coverage_level']).size().reset_index(name='count')
        sunburst_fig = px.sunburst(sunburst_data, path=['region', 'policy_type', 'coverage_level'], values='count',
                                  title="☀️ Regional Policy & Coverage Analysis",
                                  color_discrete_sequence=colors_primary)
        sunburst_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        sunburst_fig = go.Figure()
        sunburst_fig.update_layout(title="☀️ Regional Policy Analysis")
    
    # 12. Treemap Chart - Charges by region and policy
    if not filtered_df.empty:
        treemap_data = filtered_df.groupby(['region', 'policy_type'])['charges'].sum().reset_index()
        treemap_fig = px.treemap(treemap_data, path=['region', 'policy_type'], values='charges',
                                title="🌳 Total Charges by Region & Policy",
                                color_discrete_sequence=colors_gradient)
        treemap_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        treemap_fig = go.Figure()
        treemap_fig.update_layout(title="🌳 Total Charges by Region & Policy")
    
    # 13. Radar Chart - Multi-dimensional customer profile
    if not filtered_df.empty and 'education' in filtered_df.columns:
        radar_data = filtered_df.groupby('education').agg({
            'age': 'mean',
            'bmi': 'mean', 
            'charges': 'mean',
            'risk_score': 'mean',
            'credit_score': 'mean'
        }).reset_index()
        
        radar_fig = go.Figure()
        for i, education in enumerate(radar_data['education']):
            values = [radar_data.iloc[i][col] for col in ['age', 'bmi', 'charges', 'risk_score', 'credit_score']]
            # Normalize values for radar chart
            normalized_values = [(v - min(values)) / (max(values) - min(values)) * 100 for v in values]
            radar_fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=['Age', 'BMI', 'Charges', 'Risk Score', 'Credit Score'],
                fill='toself',
                name=education,
                line_color=colors_primary[i % len(colors_primary)]
            ))
        
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            title="🕸️ Customer Profile Radar Analysis",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        radar_fig = go.Figure()
        radar_fig.update_layout(title="🕸️ Customer Profile Analysis")
    
    # 14. Polar Chart - Claims distribution
    if not filtered_df.empty and 'claims_count' in filtered_df.columns:
        polar_data = filtered_df.groupby('region')['claims_count'].mean().reset_index()
        polar_fig = px.bar_polar(polar_data, r='claims_count', theta='region',
                                title="🧭 Average Claims by Region",
                                color='claims_count',
                                color_continuous_scale='Viridis')
        polar_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        polar_fig = go.Figure()
        polar_fig.update_layout(title="🧭 Average Claims by Region")
    
    # 15. Waterfall Chart - Premium vs Claims analysis
    if not filtered_df.empty and 'premium_paid' in filtered_df.columns:
        waterfall_data = filtered_df.groupby('policy_type').agg({
            'premium_paid': 'sum',
            'claim_amount': 'sum'
        }).reset_index()
        waterfall_data['net_profit'] = waterfall_data['premium_paid'] - waterfall_data['claim_amount']
        
        waterfall_fig = go.Figure(go.Waterfall(
            name="Insurance Analysis",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Basic", "Premium", "Gold", "Net Total"],
            y=[waterfall_data.iloc[0]['premium_paid'], 
               waterfall_data.iloc[1]['premium_paid'], 
               waterfall_data.iloc[2]['premium_paid'],
               waterfall_data['net_profit'].sum()],
            connector={"line":{"color":"rgb(63, 63, 63)"}},
        ))
        waterfall_fig.update_layout(
            title="💧 Premium vs Claims Waterfall Analysis",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        waterfall_fig = go.Figure()
        waterfall_fig.update_layout(title="💧 Premium vs Claims Analysis")
    
    # 16. Funnel Chart - Customer journey
    if not filtered_df.empty:
        funnel_data = filtered_df.groupby('coverage_level').size().reset_index(name='count')
        funnel_fig = px.funnel(funnel_data, x='count', y='coverage_level',
                              title="🔻 Customer Coverage Level Funnel",
                              color_discrete_sequence=colors_gradient)
        funnel_fig.update_layout(
            title_x=0.5, 
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        funnel_fig = go.Figure()
        funnel_fig.update_layout(title="🔻 Customer Coverage Funnel")
    
    # 17. Sankey Chart - Flow analysis
    if not filtered_df.empty:
        sankey_data = filtered_df.groupby(['region', 'policy_type', 'coverage_level']).size().reset_index(name='count')
        
        # Create node labels
        all_nodes = list(set(sankey_data['region'].tolist() + 
                           sankey_data['policy_type'].tolist() + 
                           sankey_data['coverage_level'].tolist()))
        node_labels = {node: i for i, node in enumerate(all_nodes)}
        
        # Create links
        links = []
        for _, row in sankey_data.iterrows():
            links.append({
                'source': node_labels[row['region']],
                'target': node_labels[row['policy_type']],
                'value': row['count']
            })
            links.append({
                'source': node_labels[row['policy_type']],
                'target': node_labels[row['coverage_level']],
                'value': row['count']
            })
        
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=colors_primary[:len(all_nodes)]
            ),
            link=dict(
                source=[link['source'] for link in links],
                target=[link['target'] for link in links],
                value=[link['value'] for link in links]
            )
        )])
        sankey_fig.update_layout(
            title="🌊 Customer Flow Analysis (Region → Policy → Coverage)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        sankey_fig = go.Figure()
        sankey_fig.update_layout(title="🌊 Customer Flow Analysis")
    
    # 18. Candlestick Chart - Monthly charges analysis
    if 'date' in filtered_df.columns and not filtered_df.empty:
        monthly_data = filtered_df.set_index('date').resample('M')['charges'].agg(['min', 'max', 'mean']).reset_index()
        monthly_data.columns = ['date', 'low', 'high', 'close']
        monthly_data['open'] = monthly_data['close'].shift(1).fillna(monthly_data['close'])
        
        candlestick_fig = go.Figure(data=go.Candlestick(
            x=monthly_data['date'],
            open=monthly_data['open'],
            high=monthly_data['high'],
            low=monthly_data['low'],
            close=monthly_data['close']
        ))
        candlestick_fig.update_layout(
            title="🕯️ Monthly Charges Candlestick Analysis",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        candlestick_fig = go.Figure()
        candlestick_fig.update_layout(title="🕯️ Monthly Charges Analysis")
    
    # 19. Contour Chart - Age vs BMI density
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['age', 'bmi']):
        contour_fig = go.Figure(data=go.Contour(
            x=filtered_df['age'],
            y=filtered_df['bmi'],
            colorscale='Viridis'
        ))
        contour_fig.update_layout(
            title="🌊 Age vs BMI Density Contour",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        contour_fig = go.Figure()
        contour_fig.update_layout(title="🌊 Age vs BMI Density")
    
    # 20. Surface Chart - 3D risk analysis
    if not filtered_df.empty and all(col in filtered_df.columns for col in ['age', 'bmi', 'risk_score']):
        # Create grid for surface plot
        age_range = np.linspace(filtered_df['age'].min(), filtered_df['age'].max(), 20)
        bmi_range = np.linspace(filtered_df['bmi'].min(), filtered_df['bmi'].max(), 20)
        age_grid, bmi_grid = np.meshgrid(age_range, bmi_range)
        
        # Calculate average risk score for each grid point
        risk_grid = np.zeros_like(age_grid)
        for i in range(len(age_range)):
            for j in range(len(bmi_range)):
                mask = (abs(filtered_df['age'] - age_range[i]) < 2) & (abs(filtered_df['bmi'] - bmi_range[j]) < 1)
                if mask.sum() > 0:
                    risk_grid[j, i] = filtered_df[mask]['risk_score'].mean()
        
        surface_fig = go.Figure(data=go.Surface(
            x=age_grid,
            y=bmi_grid,
            z=risk_grid,
            colorscale='Plasma'
        ))
        surface_fig.update_layout(
            title="🏔️ 3D Risk Score Surface (Age vs BMI)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
    else:
        surface_fig = go.Figure()
        surface_fig.update_layout(title="🏔️ 3D Risk Score Surface")
    
    # Data table
    table_data = filtered_df.head(20).to_dict('records')
    
    return (line_fig, bar_fig, scatter_fig, pie_fig, histogram_fig, 
            box_fig, heatmap_fig, bubble_fig, area_fig, violin_fig,
            sunburst_fig, treemap_fig, radar_fig, polar_fig, waterfall_fig,
            funnel_fig, sankey_fig, candlestick_fig, contour_fig, surface_fig, table_data)

# ======================
# Run app
# ======================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)


