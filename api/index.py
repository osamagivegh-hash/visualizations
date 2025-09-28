"""
VERCEL API ROUTE: Insurance Dashboard
====================================
Simple API endpoint for Vercel deployment
"""

import os
from app_vercel import app

# Export the Flask app for Vercel
application = app.server

# For Vercel serverless functions
def handler(request):
    return application(request.environ, lambda *args: None)

if __name__ == "__main__":
    app.run(debug=False)
