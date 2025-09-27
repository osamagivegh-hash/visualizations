"""
Solution 3: Callback Factory Pattern
===================================

This approach uses a factory pattern to create callbacks
dynamically based on the current state.
"""

import dash
from dash import dcc, html, Input, Output, State, callback

class CallbackFactory:
    """Factory for creating callbacks dynamically."""
    
    def __init__(self, app):
        self.app = app
        self.registered_callbacks = set()
    
    def create_auth_callback(self):
        """Create authentication callback."""
        @self.app.callback(
            [Output('auth-state', 'data'),
             Output('main-content', 'children')],
            [Input('login-btn', 'n_clicks'),
             Input('logout-btn', 'n_clicks')],
            [State('username', 'value'),
             State('password', 'value'),
             State('auth-state', 'data')]
        )
        def handle_auth(login_clicks, logout_clicks, username, password, auth_state):
            ctx = dash.callback_context
            
            if not ctx.triggered:
                return auth_state, self.get_auth_layout()
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'logout-btn':
                return {'authenticated': False}, self.get_auth_layout()
            
            elif trigger_id == 'login-btn' and login_clicks:
                if username == 'admin' and password == 'admin123':
                    return {'authenticated': True}, self.get_dashboard_layout()
            
            return auth_state, self.get_auth_layout()
    
    def create_chart_callback(self):
        """Create chart callback."""
        @self.app.callback(
            Output('chart', 'figure'),
            [Input('auth-state', 'data')]
        )
        def update_chart(auth_state):
            if auth_state.get('authenticated'):
                import plotly.graph_objects as go
                fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
                return fig
            return {}
    
    def get_auth_layout(self):
        """Get authentication layout."""
        return html.Div([
            html.H1("Login"),
            dcc.Input(id='username', type='text'),
            dcc.Input(id='password', type='password'),
            html.Button('Login', id='login-btn')
        ])
    
    def get_dashboard_layout(self):
        """Get dashboard layout."""
        return html.Div([
            html.H1("Dashboard"),
            html.Button('Logout', id='logout-btn'),
            dcc.Graph(id='chart')
        ])

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Create factory and register callbacks
factory = CallbackFactory(app)
factory.create_auth_callback()
factory.create_chart_callback()

# App layout
app.layout = html.Div([
    dcc.Store(id='auth-state', data={'authenticated': False}),
    html.Div(id='main-content', children=factory.get_auth_layout())
])

if __name__ == '__main__':
    app.run(debug=True)




