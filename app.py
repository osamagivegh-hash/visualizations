"""
Enhanced Interactive Insurance Analytics Dashboard
================================================

A comprehensive insurance analytics dashboard with authentication, multiple visualizations,
and advanced filtering capabilities.

Features:
- Basic authentication (admin/admin123)
- 8 different visualization types
- Advanced filtering with dropdowns, date picker, and sliders
- MySQL database integration with error handling
- Responsive design with Bootstrap

Author: Python 4 Assistant
Date: 2024
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import mysql.connector
from mysql.connector import Error
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import logging
import numpy as np
from scipy import stats
import base64
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'insurance_db',
    'user': 'Osama',
    'password': 'YourPassword123',
    'port': 3306,
    'charset': 'utf8mb4',
    'autocommit': True
}

# Authentication configuration
AUTH_CONFIG = {
    'username': 'admin',
    'password': 'admin123'
}

class DatabaseManager:
    """Handles database connections and operations with error handling."""
    
    def __init__(self, config):
        self.config = config
        self.connection = None
    
    def connect(self):
        """Establish database connection with error handling."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def execute_query(self, query, params=None):
        """Execute SQL query and return results."""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Error as e:
            logger.error(f"Error executing query: {e}")
            return None

def load_insurance_data():
    """Load and clean insurance data from MySQL database."""
    db_manager = DatabaseManager(DB_CONFIG)
    
    # Check what tables are available
    tables_query = "SHOW TABLES"
    tables = db_manager.execute_query(tables_query)
    
    if not tables:
        logger.error("No tables found in database or connection failed")
        return None
    
    db_name = DB_CONFIG['database']
    logger.info(f"Available tables: {[table[f'Tables_in_{db_name}'] for table in tables]}")
    
    # Try to find a suitable table for insurance analytics
    possible_tables = ['claims', 'policies', 'customers', 'insurance_data', 'transactions', 'insurance']
    selected_table = None
    
    for table_info in tables:
        table_name = table_info[f'Tables_in_{db_name}']
        if any(possible_name in table_name.lower() for possible_name in possible_tables):
            selected_table = table_name
            break
    
    if not selected_table:
        selected_table = tables[0][f'Tables_in_{db_name}']
    
    logger.info(f"Using table: {selected_table}")
    
    # Get table structure
    structure_query = f"DESCRIBE {selected_table}"
    structure = db_manager.execute_query(structure_query)
    
    if structure:
        logger.info(f"Table structure: {structure}")
    
    # Load data from the selected table
    data_query = f"SELECT * FROM {selected_table}"
    data = db_manager.execute_query(data_query)
    
    db_manager.disconnect()
    
    if not data:
        logger.error("No data retrieved from database")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Basic data cleaning
    df = clean_data(df)
    
    return df, selected_table

def clean_data(df):
    """Perform comprehensive data cleaning operations."""
    logger.info(f"Original data shape: {df.shape}")
    
    # Handle missing values
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if not mode_value.empty:
                df[col].fillna(mode_value[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    # Convert date columns if they exist
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Add synthetic date column if none exists for time series analysis
    if not any('date' in col.lower() for col in df.columns):
        df['synthetic_date'] = pd.date_range('2023-01-01', periods=len(df), freq='D')
    
    logger.info(f"Cleaned data shape: {df.shape}")
    return df

def create_line_chart(df, date_col=None, value_col=None):
    """Create a line chart showing trends over time."""
    if df.empty:
        return go.Figure()
    
    # Find date and value columns
    if not date_col:
        date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        date_col = date_candidates[0] if date_candidates else 'synthetic_date'
    
    if not value_col:
        numeric_cols = df.select_dtypes(include=['number']).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if not value_col:
        return go.Figure()
    
    # Group by date and aggregate
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df_grouped = df.groupby(df[date_col].dt.date)[value_col].mean().reset_index()
    else:
        df_grouped = df.groupby(date_col)[value_col].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_grouped[date_col],
        y=df_grouped[value_col],
        mode='lines+markers',
        name='Trend',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Trend Analysis Over Time",
        xaxis_title="Date",
        yaxis_title=value_col.replace('_', ' ').title(),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_bar_chart(df, category_col=None, value_col=None):
    """Create a bar chart for categorical comparisons."""
    if df.empty:
        return go.Figure()
    
    if not category_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        category_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    if not value_col:
        numeric_cols = df.select_dtypes(include=['number']).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if not category_col or not value_col:
        return go.Figure()
    
    # Group by category and aggregate
    df_grouped = df.groupby(category_col)[value_col].mean().reset_index()
    df_grouped = df_grouped.sort_values(value_col, ascending=False).head(10)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_grouped[category_col],
        y=df_grouped[value_col],
        name='Comparison',
        marker_color='#ff7f0e',
        text=df_grouped[value_col].round(2),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Category Comparison",
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title(),
        xaxis_tickangle=-45,
        template='plotly_white'
    )
    
    return fig

def create_scatter_plot(df, x_col=None, y_col=None, color_col=None):
    """Create a scatter plot to show relationships between numeric variables."""
    if df.empty:
        return go.Figure()
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not x_col and len(numeric_cols) > 0:
        x_col = numeric_cols[0]
    if not y_col and len(numeric_cols) > 1:
        y_col = numeric_cols[1]
    
    if not x_col or not y_col:
        return go.Figure()
    
    if not color_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    fig = go.Figure()
    
    if color_col:
        unique_categories = df[color_col].unique()[:8]
        colors = px.colors.qualitative.Set3
        
        for i, category in enumerate(unique_categories):
            df_filtered = df[df[color_col] == category]
            fig.add_trace(go.Scatter(
                x=df_filtered[x_col],
                y=df_filtered[y_col],
                mode='markers',
                name=str(category),
                marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Data Points',
            marker=dict(color='#2ca02c', size=8, opacity=0.7)
        ))
    
    fig.update_layout(
        title="Relationship Analysis",
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

def create_pie_chart(df, category_col=None, value_col=None):
    """Create a pie chart showing percentage share."""
    if df.empty:
        return go.Figure()
    
    if not category_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        category_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    if not value_col:
        numeric_cols = df.select_dtypes(include=['number']).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if not category_col or not value_col:
        return go.Figure()
    
    # Group by category and aggregate
    df_grouped = df.groupby(category_col)[value_col].sum().reset_index()
    df_grouped = df_grouped.sort_values(value_col, ascending=False)
    
    fig = go.Figure(data=[go.Pie(
        labels=df_grouped[category_col],
        values=df_grouped[value_col],
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Distribution by Category",
        template='plotly_white'
    )
    
    return fig

def create_histogram(df, column=None):
    """Create a histogram showing distribution."""
    if df.empty:
        return go.Figure()
    
    if not column:
        numeric_cols = df.select_dtypes(include=['number']).columns
        column = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if not column:
        return go.Figure()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[column],
        nbinsx=30,
        marker_color='#9467bd',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"Distribution of {column.replace('_', ' ').title()}",
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title="Frequency",
        template='plotly_white'
    )
    
    return fig

def create_box_plot(df, category_col=None, value_col=None):
    """Create a box plot showing summary statistics."""
    if df.empty:
        return go.Figure()
    
    if not category_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        category_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    if not value_col:
        numeric_cols = df.select_dtypes(include=['number']).columns
        value_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    
    if not category_col or not value_col:
        return go.Figure()
    
    fig = go.Figure()
    
    unique_categories = df[category_col].unique()[:8]
    colors = px.colors.qualitative.Set2
    
    for i, category in enumerate(unique_categories):
        df_filtered = df[df[category_col] == category]
        fig.add_trace(go.Box(
            y=df_filtered[value_col],
            name=str(category),
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title="Summary Statistics by Category",
        xaxis_title=category_col.replace('_', ' ').title(),
        yaxis_title=value_col.replace('_', ' ').title(),
        template='plotly_white'
    )
    
    return fig

def create_heatmap(df):
    """Create a heatmap showing correlation between numeric variables."""
    if df.empty:
        return go.Figure()
    
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return go.Figure()
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        template='plotly_white'
    )
    
    return fig

def create_bubble_chart(df, x_col=None, y_col=None, size_col=None, color_col=None):
    """Create a bubble chart with size based on another variable."""
    if df.empty:
        return go.Figure()
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not x_col and len(numeric_cols) > 0:
        x_col = numeric_cols[0]
    if not y_col and len(numeric_cols) > 1:
        y_col = numeric_cols[1]
    if not size_col and len(numeric_cols) > 2:
        size_col = numeric_cols[2]
    
    if not x_col or not y_col or not size_col:
        return go.Figure()
    
    if not color_col:
        categorical_cols = df.select_dtypes(include=['object']).columns
        color_col = categorical_cols[0] if len(categorical_cols) > 0 else None
    
    fig = go.Figure()
    
    if color_col:
        unique_categories = df[color_col].unique()[:6]
        colors = px.colors.qualitative.Set1
        
        for i, category in enumerate(unique_categories):
            df_filtered = df[df[color_col] == category]
            fig.add_trace(go.Scatter(
                x=df_filtered[x_col],
                y=df_filtered[y_col],
                mode='markers',
                name=str(category),
                marker=dict(
                    size=df_filtered[size_col] / df_filtered[size_col].max() * 30 + 5,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                )
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            name='Data Points',
            marker=dict(
                size=df[size_col] / df[size_col].max() * 30 + 5,
                color='#17becf',
                opacity=0.7,
                line=dict(width=1, color='white')
            )
        ))
    
    fig.update_layout(
        title="Bubble Chart Analysis",
        xaxis_title=x_col.replace('_', ' ').title(),
        yaxis_title=y_col.replace('_', ' ').title(),
        hovermode='closest',
        template='plotly_white'
    )
    
    return fig

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Enhanced Insurance Analytics Dashboard"

# Simple authentication check
def check_auth(username, password):
    """Simple authentication check."""
    logger.info(f"Checking auth: {username} == {AUTH_CONFIG['username']}, {password} == {AUTH_CONFIG['password']}")
    result = username == AUTH_CONFIG['username'] and password == AUTH_CONFIG['password']
    logger.info(f"Auth result: {result}")
    return result

# Load data
logger.info("Loading data from database...")
data_result = load_insurance_data()

if data_result is None:
    logger.warning("Database connection failed, using sample data")
    np.random.seed(42)
    sample_data = {
        'age': np.random.randint(18, 65, 1000),
        'sex': np.random.choice(['male', 'female'], 1000),
        'bmi': np.random.normal(28, 6, 1000),
        'children': np.random.randint(0, 6, 1000),
        'smoker': np.random.choice(['yes', 'no'], 1000),
        'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], 1000),
        'charges': np.random.normal(13000, 12000, 1000)
    }
    df = pd.DataFrame(sample_data)
    df['charges'] = np.abs(df['charges'])  # Ensure positive charges
    table_name = "sample_data"
else:
    df, table_name = data_result

# Authentication layout
auth_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Insurance Analytics Dashboard", className="text-center mb-4"),
            html.Hr(),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Login Required", className="text-center mb-4"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Username"),
                        dbc.Input(id="username-input", type="text", placeholder="Enter username")
                    ], className="mb-3"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Password"),
                        dbc.Input(id="password-input", type="password", placeholder="Enter password")
                    ], className="mb-3"),
                    dbc.Button("Login", id="login-button", color="primary", className="w-100"),
                    html.Div(id="auth-message", className="mt-3")
                ])
            ], style={"maxWidth": "400px", "margin": "0 auto"})
        ], width=12)
    ])
], fluid=True)

# Main dashboard layout
dashboard_layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Enhanced Insurance Analytics Dashboard", className="text-center mb-4"),
            html.Hr(),
            dbc.Button("Logout", id="logout-button", color="danger", size="sm", className="float-end")
        ])
    ]),
    
    # Data info
    dbc.Row([
        dbc.Col([
            html.H4(f"Data Source: {table_name}"),
            html.P(f"Total Records: {len(df)}"),
            html.P(f"Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        ], width=12)
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': str(val), 'value': str(val)} for val in df.select_dtypes(include=['object']).iloc[:, 0].unique()[:10]],
                value='All',
                clearable=False
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df.select_dtypes(include=['datetime']).iloc[:, 0].min() if len(df.select_dtypes(include=['datetime']).columns) > 0 else datetime.now() - timedelta(days=30),
                end_date=df.select_dtypes(include=['datetime']).iloc[:, 0].max() if len(df.select_dtypes(include=['datetime']).columns) > 0 else datetime.now(),
                display_format='YYYY-MM-DD'
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Age Range:"),
            dcc.RangeSlider(
                id='age-slider',
                min=df['age'].min() if 'age' in df.columns else 0,
                max=df['age'].max() if 'age' in df.columns else 100,
                value=[df['age'].min() if 'age' in df.columns else 0, df['age'].max() if 'age' in df.columns else 100],
                marks={i: str(i) for i in range(int(df['age'].min() if 'age' in df.columns else 0), int(df['age'].max() if 'age' in df.columns else 100) + 1, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Refresh Data:"),
            html.Br(),
            dbc.Button("Refresh", id="refresh-button", color="primary", size="sm")
        ], width=3)
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-chart')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='bar-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter-plot')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='pie-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 3
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='histogram')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='box-plot')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 4
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='bubble-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Data table
    dbc.Row([
        dbc.Col([
            html.H4("Data Preview"),
            dash_table.DataTable(
                id='data-table',
                data=df.head(20).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ], width=12)
    ], className="mt-4")
    
], fluid=True)

# App layout
app.layout = html.Div([
    dcc.Store(id='auth-status', data={'authenticated': False}),
    html.Div(id='main-content', children=auth_layout)
])

# SOLUTION: CENTRALIZED STATE MANAGEMENT WITH DCC.STORE
# =============================================================================

# Authentication layout
auth_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Insurance Analytics Dashboard", className="text-center mb-4"),
            html.Hr(),
            dbc.Card([
                dbc.CardBody([
                    html.H4("Login Required", className="text-center mb-4"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Username"),
                        dbc.Input(id="username-input", type="text", placeholder="Enter username")
                    ], className="mb-3"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Password"),
                        dbc.Input(id="password-input", type="password", placeholder="Enter password")
                    ], className="mb-3"),
                    dbc.Button("Login", id="login-button", color="primary", className="w-100"),
                    html.Div(id="auth-message", className="mt-3")
                ])
            ], style={"maxWidth": "400px", "margin": "0 auto"})
        ], width=12)
    ])
], fluid=True)

# Dashboard layout
dashboard_layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Enhanced Insurance Analytics Dashboard", className="text-center mb-4"),
            html.Hr(),
            dbc.Button("Logout", id="logout-button", color="danger", size="sm", className="float-end")
        ])
    ]),
    
    # Data info
    dbc.Row([
        dbc.Col([
            html.H4(f"Data Source: {table_name}"),
            html.P(f"Total Records: {len(df)}"),
            html.P(f"Data Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        ], width=12)
    ], className="mb-4"),
    
    # Filters
    dbc.Row([
        dbc.Col([
            html.Label("Select Category:"),
            dcc.Dropdown(
                id='category-filter',
                options=[{'label': 'All', 'value': 'All'}] + 
                        [{'label': str(val), 'value': str(val)} for val in df.select_dtypes(include=['object']).iloc[:, 0].unique()[:10]],
                value='All',
                clearable=False
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=df.select_dtypes(include=['datetime']).iloc[:, 0].min() if len(df.select_dtypes(include=['datetime']).columns) > 0 else datetime.now() - timedelta(days=30),
                end_date=df.select_dtypes(include=['datetime']).iloc[:, 0].max() if len(df.select_dtypes(include=['datetime']).columns) > 0 else datetime.now(),
                display_format='YYYY-MM-DD'
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Age Range:"),
            dcc.RangeSlider(
                id='age-slider',
                min=df['age'].min() if 'age' in df.columns else 0,
                max=df['age'].max() if 'age' in df.columns else 100,
                value=[df['age'].min() if 'age' in df.columns else 0, df['age'].max() if 'age' in df.columns else 100],
                marks={i: str(i) for i in range(int(df['age'].min() if 'age' in df.columns else 0), int(df['age'].max() if 'age' in df.columns else 100) + 1, 10)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=3),
        
        dbc.Col([
            html.Label("Refresh Data:"),
            html.Br(),
            dbc.Button("Refresh", id="refresh-button", color="primary", size="sm")
        ], width=3)
    ], className="mb-4"),
    
    # Charts Row 1
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-chart')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='bar-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 2
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='scatter-plot')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='pie-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 3
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='histogram')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='box-plot')
        ], width=6)
    ], className="mb-4"),
    
    # Charts Row 4
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='heatmap')
        ], width=6),
        
        dbc.Col([
            dcc.Graph(id='bubble-chart')
        ], width=6)
    ], className="mb-4"),
    
    # Data table
    dbc.Row([
        dbc.Col([
            html.H4("Data Preview"),
            dash_table.DataTable(
                id='data-table',
                data=df.head(20).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=10,
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ], width=12)
    ], className="mt-4")
    
], fluid=True)

# =============================================================================
# CENTRALIZED STATE MANAGEMENT LAYOUT
# =============================================================================

app.layout = html.Div([
    # Centralized state stores
    dcc.Store(id='auth-store', data={'authenticated': False, 'username': None}),
    dcc.Store(id='ui-store', data={'current_layout': 'auth', 'message': ''}),
    dcc.Store(id='data-store', data=df.to_dict('records')),
    
    # Main content area
    html.Div(id='main-content', children=auth_layout)
])

# =============================================================================
# SOLUTION: SINGLE AUTHENTICATION CALLBACK
# =============================================================================

@app.callback(
    [Output('auth-store', 'data'),
     Output('ui-store', 'data')],
    [Input('login-button', 'n_clicks'),
     Input('logout-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('password-input', 'value'),
     State('auth-store', 'data')]
)
def handle_authentication(login_clicks, logout_clicks, username, password, auth_data):
    """
    Single callback handling authentication state changes.
    This prevents duplicate callback outputs by having only one callback
    manage the authentication state.
    """
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return auth_data, {'current_layout': 'auth', 'message': ''}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'logout-button' and logout_clicks:
        logger.info("Logout triggered")
        return {'authenticated': False, 'username': None}, {'current_layout': 'auth', 'message': ''}
    
    elif trigger_id == 'login-button' and login_clicks:
        logger.info(f"Login attempt: username={username}")
        
        if check_auth(username, password):
            logger.info("Authentication successful")
            return {'authenticated': True, 'username': username}, {'current_layout': 'dashboard', 'message': ''}
        else:
            logger.info("Authentication failed")
            return auth_data, {'current_layout': 'auth', 'message': 'Invalid username or password'}
    
    return auth_data, {'current_layout': 'auth', 'message': ''}

# =============================================================================
# SOLUTION: LAYOUT RENDERING CALLBACK
# =============================================================================

@app.callback(
    [Output('main-content', 'children'),
     Output('auth-message', 'children')],
    [Input('ui-store', 'data')],
    [State('auth-store', 'data')]
)
def render_layout(ui_data, auth_data):
    """
    Single callback responsible for rendering the appropriate layout.
    This prevents duplicate outputs by having only one callback
    manage the main content area.
    """
    current_layout = ui_data.get('current_layout', 'auth')
    message = ui_data.get('message', '')
    
    if current_layout == 'dashboard' and auth_data.get('authenticated', False):
        # Show dashboard
        return dashboard_layout, ""
    else:
        # Show auth layout with message
        if message:
            message_component = dbc.Alert(message, color="danger")
        else:
            message_component = ""
        
        # Update auth layout with message
        updated_auth_layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Insurance Analytics Dashboard", className="text-center mb-4"),
                    html.Hr(),
                    message_component,
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Login Required", className="text-center mb-4"),
                            dbc.InputGroup([
                                dbc.InputGroupText("Username"),
                                dbc.Input(id="username-input", type="text", placeholder="Enter username")
                            ], className="mb-3"),
                            dbc.InputGroup([
                                dbc.InputGroupText("Password"),
                                dbc.Input(id="password-input", type="password", placeholder="Enter password")
                            ], className="mb-3"),
                            dbc.Button("Login", id="login-button", color="primary", className="w-100"),
                            html.Div(id="auth-message", className="mt-3")
                        ])
                    ], style={"maxWidth": "400px", "margin": "0 auto"})
                ], width=12)
            ])
        ], fluid=True)
        
        return updated_auth_layout, ""

# Chart update callbacks
@app.callback(
    [Output('line-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('pie-chart', 'figure'),
     Output('histogram', 'figure'),
     Output('box-plot', 'figure'),
     Output('heatmap', 'figure'),
     Output('bubble-chart', 'figure'),
     Output('data-table', 'data')],
    [Input('category-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('age-slider', 'value'),
     Input('refresh-button', 'n_clicks')]
)
def update_charts(category_filter, start_date, end_date, age_range, refresh_clicks):
    """Update all charts based on filter selections."""
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    # Apply category filter
    if category_filter != 'All':
        categorical_col = df.select_dtypes(include=['object']).columns[0]
        filtered_df = filtered_df[filtered_df[categorical_col] == category_filter]
    
    # Apply date filter if date column exists
    date_cols = df.select_dtypes(include=['datetime']).columns
    if len(date_cols) > 0 and start_date and end_date:
        date_col = date_cols[0]
        filtered_df = filtered_df[
            (filtered_df[date_col] >= start_date) & 
            (filtered_df[date_col] <= end_date)
        ]
    
    # Apply age filter
    if 'age' in filtered_df.columns and age_range:
        filtered_df = filtered_df[
            (filtered_df['age'] >= age_range[0]) & 
            (filtered_df['age'] <= age_range[1])
        ]
    
    # Create all charts
    line_fig = create_line_chart(filtered_df)
    bar_fig = create_bar_chart(filtered_df)
    scatter_fig = create_scatter_plot(filtered_df)
    pie_fig = create_pie_chart(filtered_df)
    histogram_fig = create_histogram(filtered_df)
    box_fig = create_box_plot(filtered_df)
    heatmap_fig = create_heatmap(filtered_df)
    bubble_fig = create_bubble_chart(filtered_df)
    
    # Update data table
    table_data = filtered_df.head(20).to_dict('records')
    
    return line_fig, bar_fig, scatter_fig, pie_fig, histogram_fig, box_fig, heatmap_fig, bubble_fig, table_data

if __name__ == '__main__':
    logger.info("Starting Enhanced Insurance Analytics Dashboard...")
    app.run(debug=True, host='0.0.0.0', port=8050)