"""
MAIN APP: Insurance Dashboard for Vercel
=======================================
Simplified version for better Vercel compatibility
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Create sample data
def create_sample_data():
    np.random.seed(42)
    n_records = 1000
    
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
        'region': regions,
        'date': pd.date_range('2020-01-01', periods=n_records, freq='D')
    })
    
    return df

# Load data
df = create_sample_data()

# Create Dash app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True)

app.title = "🚀 Insurance Analytics Dashboard"

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

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚀 Advanced Insurance Analytics Dashboard", 
                   className="text-center mb-4 text-white")
        ], width=12)
    ]),
    
    # Key metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{len(df):,}", className="text-center text-white"),
                    html.P("Total Records", className="text-center text-white-50")
                ])
            ], color="primary", className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"${df['charges'].mean():,.0f}", className="text-center text-white"),
                    html.P("Avg Charges", className="text-center text-white-50")
                ])
            ], color="success", className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{df['age'].mean():.1f}", className="text-center text-white"),
                    html.P("Avg Age", className="text-center text-white-50")
                ])
            ], color="info", className="text-center")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{df['bmi'].mean():.1f}", className="text-center text-white"),
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
    
    # ML Predictions
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("🤖 AI Insurance Predictions", className="text-center text-white mb-4"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Age:", className="text-white"),
                            dbc.Input(id="ml-age", type="number", value=30, min=18, max=80)
                        ], width=3),
                        dbc.Col([
                            html.Label("BMI:", className="text-white"),
                            dbc.Input(id="ml-bmi", type="number", value=25, min=15, max=50, step=0.1)
                        ], width=3),
                        dbc.Col([
                            html.Label("Children:", className="text-white"),
                            dbc.Input(id="ml-children", type="number", value=0, min=0, max=5)
                        ], width=3),
                        dbc.Col([
                            dbc.Button("🔮 PREDICT", id="predict-button", color="success", className="mt-4")
                        ], width=3)
                    ], className="mb-4"),
                    html.Div(id="prediction-results", className="text-center")
                ])
            ], style={'backgroundColor': 'rgba(0,0,0,0.8)', 'border': '3px solid #00FF00'})
        ], width=12)
    ], className="mb-4")
], fluid=True)

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
    line_data = df.groupby('date')['charges'].mean().reset_index()
    line_fig = px.line(line_data, x='date', y='charges', title="📈 Insurance Charges Trends")
    
    # Bar Chart
    bar_data = df.groupby('region')['charges'].mean().reset_index()
    bar_fig = px.bar(bar_data, x='region', y='charges', title="📊 Average Charges by Region")
    
    # Scatter Plot
    scatter_fig = px.scatter(df, x='age', y='charges', color='smoker', title="🎯 Age vs Charges Analysis")
    
    # Pie Chart
    pie_data = df['region'].value_counts().reset_index()
    pie_data.columns = ['region', 'count']
    pie_fig = px.pie(pie_data, names='region', values='count', title="🥧 Region Distribution")
    
    # Apply dark theme
    for fig in [line_fig, bar_fig, scatter_fig, pie_fig]:
        fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    
    return line_fig, bar_fig, scatter_fig, pie_fig

@app.callback(
    Output('prediction-results', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('ml-age', 'value'), State('ml-bmi', 'value'), State('ml-children', 'value')]
)
def predict_insurance(n_clicks, age, bmi, children):
    if n_clicks is None:
        return ""
    
    try:
        if age is None or bmi is None or children is None:
            return dbc.Alert("Please fill in all fields", color="warning")
        
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
            risk_color = "danger"
        elif risk_score >= 2:
            risk_level = "Medium"
            risk_color = "warning"
        else:
            risk_level = "Low"
            risk_color = "success"
        
        return dbc.Card([
            dbc.CardBody([
                html.H6("🎯 Predicted Results:", className="text-white mb-3"),
                html.P(f"💰 Charges: ${predicted_charges:,.0f}", className="text-success mb-2"),
                html.P(f"⚠️ Risk Level: {risk_level}", className=f"text-{risk_color} mb-0")
            ])
        ], style={'backgroundColor': 'rgba(0,255,0,0.1)', 'border': '1px solid rgba(0,255,0,0.3)'})
        
    except Exception as e:
        return dbc.Alert(f"❌ Prediction error: {str(e)}", color="danger")

# Server for Vercel
server = app.server

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
