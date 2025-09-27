"""
Solution 2: Conditional Callback Registration
=============================================

This approach registers callbacks conditionally to avoid
referencing non-existent components.
"""

import dash
from dash import dcc, html, Input, Output, State, callback

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Layout with all possible components (hidden when not needed)
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    
    # Auth components (always present)
    html.Div(id='auth-section', children=[
        html.H1("Login"),
        dcc.Input(id='username', type='text'),
        dcc.Input(id='password', type='password'),
        html.Button('Login', id='login-btn')
    ]),
    
    # Dashboard components (always present but hidden)
    html.Div(id='dashboard-section', style={'display': 'none'}, children=[
        html.H1("Dashboard"),
        html.Button('Logout', id='logout-btn'),
        dcc.Graph(id='chart')
    ])
])

# Single callback that handles both login and logout
@app.callback(
    [Output('auth-state', 'data'),
     Output('auth-section', 'style'),
     Output('dashboard-section', 'style')],
    [Input('login-btn', 'n_clicks'),
     Input('logout-btn', 'n_clicks')],
    [State('username', 'value'),
     State('password', 'value'),
     State('auth-state', 'data')]
)
def handle_auth(login_clicks, logout_clicks, username, password, auth_state):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return auth_state, {'display': 'block'}, {'display': 'none'}
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'logout-btn':
        return {'authenticated': False}, {'display': 'block'}, {'display': 'none'}
    
    elif trigger_id == 'login-btn' and login_clicks:
        if username == 'admin' and password == 'admin123':
            return {'authenticated': True}, {'display': 'none'}, {'display': 'block'}
    
    return auth_state, {'display': 'block'}, {'display': 'none'}

# Chart callback (only runs when dashboard is visible)
@app.callback(
    Output('chart', 'figure'),
    [Input('auth-state', 'data')]
)
def update_chart(auth_state):
    if auth_state.get('authenticated'):
        # Create chart only when authenticated
        import plotly.graph_objects as go
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
        return fig
    return {}

if __name__ == '__main__':
    app.run(debug=True)




