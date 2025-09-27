"""
Dash Best Practices for Avoiding Callback Conflicts
==================================================

This module demonstrates best practices for building
robust Dash applications without callback conflicts.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashAppBuilder:
    """
    Builder class for creating Dash apps with proper callback management.
    """
    
    def __init__(self, app_name: str):
        self.app = dash.Dash(app_name, suppress_callback_exceptions=True)
        self.callback_registry = {}
        self.state_stores = {}
        
    def add_state_store(self, store_id: str, initial_data: Any = None):
        """Add a state store to manage application state."""
        self.state_stores[store_id] = initial_data
        logger.info(f"Added state store: {store_id}")
        
    def register_callback(self, callback_id: str, outputs: List, inputs: List, 
                         states: List = None, prevent_initial_call: bool = False):
        """Register a callback with conflict checking."""
        
        # Check for conflicts
        for output in outputs:
            if output in self.callback_registry:
                raise ValueError(f"Output {output} already registered in callback {self.callback_registry[output]}")
            self.callback_registry[output] = callback_id
        
        # Create callback decorator
        def callback_decorator(func):
            return self.app.callback(
                outputs, inputs, states, prevent_initial_call=prevent_initial_call
            )(func)
        
        return callback_decorator
    
    def build_layout(self) -> html.Div:
        """Build the main layout with all state stores."""
        children = []
        
        # Add state stores
        for store_id, initial_data in self.state_stores.items():
            children.append(dcc.Store(id=store_id, data=initial_data))
        
        # Add main content area
        children.append(html.Div(id='main-content'))
        
        return html.Div(children)

# Example usage
def create_robust_dash_app():
    """Create a robust Dash app using best practices."""
    
    # Initialize builder
    builder = DashAppBuilder("RobustApp")
    
    # Add state stores
    builder.add_state_store('auth-state', {'authenticated': False})
    builder.add_state_store('ui-state', {'current_page': 'login'})
    builder.add_state_store('data-state', {})
    
    # Set layout
    builder.app.layout = builder.build_layout()
    
    # Register callbacks with conflict checking
    @builder.register_callback(
        'auth-callback',
        [Output('auth-state', 'data'), Output('ui-state', 'data')],
        [Input('login-btn', 'n_clicks'), Input('logout-btn', 'n_clicks')],
        [State('username', 'value'), State('password', 'value')]
    )
    def handle_authentication(login_clicks, logout_clicks, username, password):
        """Handle authentication state changes."""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            return {'authenticated': False}, {'current_page': 'login'}
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'logout-btn':
            return {'authenticated': False}, {'current_page': 'login'}
        
        elif trigger_id == 'login-btn' and login_clicks:
            if username == 'admin' and password == 'admin123':
                return {'authenticated': True}, {'current_page': 'dashboard'}
        
        return {'authenticated': False}, {'current_page': 'login'}
    
    @builder.register_callback(
        'layout-callback',
        [Output('main-content', 'children')],
        [Input('ui-state', 'data')]
    )
    def render_layout(ui_state):
        """Render the appropriate layout based on UI state."""
        current_page = ui_state.get('current_page', 'login')
        
        if current_page == 'dashboard':
            return html.Div([
                html.H1("Dashboard"),
                html.Button('Logout', id='logout-btn'),
                dcc.Graph(id='chart')
            ])
        else:
            return html.Div([
                html.H1("Login"),
                dcc.Input(id='username', type='text'),
                dcc.Input(id='password', type='password'),
                html.Button('Login', id='login-btn')
            ])
    
    return builder.app

# Testing utilities
def test_callback_conflicts(app):
    """Test for callback conflicts in a Dash app."""
    outputs = []
    for callback_id, callback_info in app.callback_map.items():
        outputs.extend(callback_info['outputs'])
    
    if len(outputs) != len(set(outputs)):
        raise ValueError("Duplicate callback outputs found!")
    
    logger.info("No callback conflicts detected")

# Example of proper error handling
def safe_callback_execution(func):
    """Decorator for safe callback execution with error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            # Return safe default values
            return None
    return wrapper

if __name__ == '__main__':
    # Create and test the app
    app = create_robust_dash_app()
    test_callback_conflicts(app)
    app.run(debug=True)




