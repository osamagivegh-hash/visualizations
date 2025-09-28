"""
ULTRA SIMPLE: Insurance Dashboard for Vercel
============================================
Minimal dependencies for Vercel deployment
"""

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Create sample data
def create_sample_data():
    np.random.seed(42)
    n_records = 500
    
    ages = np.random.randint(18, 80, n_records)
    bmi = 20 + (ages - 18) * 0.3 + np.random.normal(0, 5, n_records)
    bmi = np.clip(bmi, 15, 50)
    
    smoker = np.random.binomial(1, 0.2, n_records)
    base_charges = 2000 + ages * 50 + bmi * 100 + smoker * 15000
    charges = base_charges + np.random.normal(0, 3000, n_records)
    charges = np.clip(charges, 1000, 50000)
    
    children = np.random.poisson(1.5, n_records)
    children = np.clip(children, 0, 5)
    charges += children * 2000
    
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_records)
    
    df = pd.DataFrame({
        'age': ages,
        'sex': np.random.choice(['male', 'female'], n_records),
        'bmi': bmi,
        'children': children,
        'smoker': ['yes' if s else 'no' for s in smoker],
        'charges': charges,
        'region': regions
    })
    
    return df

# Load data
df = create_sample_data()

# Create Dash app
app = dash.Dash(__name__)

app.title = "🚀 Insurance Analytics Dashboard"

# Simple layout
app.layout = html.Div([
    html.H1("🚀 Advanced Insurance Analytics Dashboard", 
            style={'textAlign': 'center', 'color': 'white', 'marginBottom': '30px'}),
    
    # Key metrics
    html.Div([
        html.Div([
            html.H3(f"{len(df):,}", style={'color': 'white', 'textAlign': 'center'}),
            html.P("Total Records", style={'color': 'white', 'textAlign': 'center'})
        ], style={'backgroundColor': 'rgba(0,123,255,0.8)', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H3(f"${df['charges'].mean():,.0f}", style={'color': 'white', 'textAlign': 'center'}),
            html.P("Avg Charges", style={'color': 'white', 'textAlign': 'center'})
        ], style={'backgroundColor': 'rgba(40,167,69,0.8)', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H3(f"{df['age'].mean():.1f}", style={'color': 'white', 'textAlign': 'center'}),
            html.P("Avg Age", style={'color': 'white', 'textAlign': 'center'})
        ], style={'backgroundColor': 'rgba(23,162,184,0.8)', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'}),
        
        html.Div([
            html.H3(f"{df['bmi'].mean():.1f}", style={'color': 'white', 'textAlign': 'center'}),
            html.P("Avg BMI", style={'color': 'white', 'textAlign': 'center'})
        ], style={'backgroundColor': 'rgba(255,193,7,0.8)', 'padding': '20px', 'borderRadius': '10px', 'margin': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    
    # Charts
    html.Div([
        dcc.Graph(id='line-chart'),
        dcc.Graph(id='bar-chart')
    ], style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    html.Div([
        dcc.Graph(id='scatter-plot'),
        dcc.Graph(id='pie-chart')
    ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginTop': '20px'}),
    
    # ML Predictions
    html.Div([
        html.H2("🤖 AI Insurance Predictions", style={'color': 'white', 'textAlign': 'center', 'marginTop': '40px'}),
        html.Div([
            html.Label("Age:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.Input(id="ml-age", type="number", value=30, min=18, max=80, style={'marginRight': '20px'}),
            
            html.Label("BMI:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.Input(id="ml-bmi", type="number", value=25, min=15, max=50, step=0.1, style={'marginRight': '20px'}),
            
            html.Label("Children:", style={'color': 'white', 'marginRight': '10px'}),
            dcc.Input(id="ml-children", type="number", value=0, min=0, max=5, style={'marginRight': '20px'}),
            
            html.Button("🔮 PREDICT", id="predict-button", n_clicks=0, 
                       style={'backgroundColor': '#28a745', 'color': 'white', 'border': 'none', 
                             'padding': '10px 20px', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        html.Div(id="prediction-results", style={'textAlign': 'center'})
    ], style={'backgroundColor': 'rgba(0,0,0,0.8)', 'padding': '30px', 'borderRadius': '15px', 
              'border': '3px solid #00FF00', 'marginTop': '30px'})
], style={'backgroundColor': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          'minHeight': '100vh', 'padding': '20px'})

# Callbacks
@app.callback(
    [Output('line-chart', 'figure'),
     Output('bar-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('predict-button', 'n_clicks')]
)
def update_charts(n_clicks):
    # Line Chart
    line_data = df.groupby('age')['charges'].mean().reset_index()
    line_fig = px.line(line_data, x='age', y='charges', title="📈 Charges by Age")
    line_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                          font_color='white', title_font_color='white')
    
    # Bar Chart
    bar_data = df.groupby('region')['charges'].mean().reset_index()
    bar_fig = px.bar(bar_data, x='region', y='charges', title="📊 Average Charges by Region")
    bar_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font_color='white', title_font_color='white')
    
    # Scatter Plot
    scatter_fig = px.scatter(df, x='age', y='charges', color='smoker', title="🎯 Age vs Charges Analysis")
    scatter_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                             font_color='white', title_font_color='white')
    
    # Pie Chart
    pie_data = df['region'].value_counts().reset_index()
    pie_data.columns = ['region', 'count']
    pie_fig = px.pie(pie_data, names='region', values='count', title="🥧 Region Distribution")
    pie_fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                         font_color='white', title_font_color='white')
    
    return line_fig, bar_fig, scatter_fig, pie_fig

@app.callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('ml-age', 'value'), State('ml-bmi', 'value'), State('ml-children', 'value')]
)
def predict_insurance(n_clicks, age, bmi, children):
    if n_clicks == 0:
        return ""
    
    try:
        if age is None or bmi is None or children is None:
            return html.Div("Please fill in all fields", style={'color': 'red', 'fontSize': '18px'})
        
        # Simple prediction logic
        base_charge = 2000 + age * 50 + bmi * 100
        base_charge += children * 2000
        predicted_charges = base_charge + np.random.normal(0, 2000)
        predicted_charges = max(predicted_charges, 1000)
        
        # Risk calculation
        risk_score = 0
        if age > 50:
            risk_score += 2
        elif age > 40:
            risk_score += 1
        if bmi > 30:
            risk_score += 3
        elif bmi > 25:
            risk_score += 1
        if children > 3:
            risk_score += 1
        
        if risk_score >= 5:
            risk_level = "High"
            risk_color = "red"
        elif risk_score >= 2:
            risk_level = "Medium"
            risk_color = "orange"
        else:
            risk_level = "Low"
            risk_color = "green"
        
        return html.Div([
            html.H4("🎯 Predicted Results:", style={'color': 'white', 'marginBottom': '20px'}),
            html.P(f"💰 Charges: ${predicted_charges:,.0f}", 
                  style={'color': '#00FF00', 'fontSize': '20px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            html.P(f"⚠️ Risk Level: {risk_level}", 
                  style={'color': risk_color, 'fontSize': '18px', 'fontWeight': 'bold'})
        ])
        
    except Exception as e:
        return html.Div(f"❌ Prediction error: {str(e)}", style={'color': 'red', 'fontSize': '18px'})

# Server for Vercel
server = app.server

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
