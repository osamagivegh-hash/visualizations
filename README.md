# Enhanced Insurance Analytics Dashboard

A comprehensive interactive dashboard for insurance analytics built with Python, Dash, and MySQL.

## Features

- **🔐 Authentication**: Basic login protection (admin/admin123)
- **🗄️ Database Integration**: Connects to MySQL database `insurance_db`
- **📊 8 Interactive Visualizations**: 
  - Line chart showing trends over time
  - Bar chart for categorical comparisons
  - Scatter plot for relationship analysis
  - Pie chart for percentage distribution
  - Histogram for data distribution
  - Box plot for summary statistics
  - Heatmap for correlation analysis
  - Bubble chart with size-based variables
- **🎛️ Advanced Filtering**: 
  - Dropdown filters for categorical variables
  - Date range picker
  - Range sliders for numeric filtering
- **🧹 Data Cleaning**: Automatic handling of missing values and data type conversion
- **🛡️ Error Handling**: Robust database connection management with fallback to sample data
- **📱 Responsive Design**: Clean, modern UI with Bootstrap components

## Prerequisites

- Python 3.8 or higher
- MySQL server running with `insurance_db` database
- Database user `Osama` with password `YourPassword123`

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure your MySQL database is running and accessible with the specified credentials.

## Usage

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
python app.py
```

3. Access the dashboard at: http://localhost:8050

4. Login with credentials:
   - Username: `admin`
   - Password: `admin123`

## Database Configuration

The application connects to MySQL using these default settings:
- Host: localhost
- Database: insurance_db
- Username: Osama
- Password: YourPassword123
- Port: 3306

To modify these settings, edit the `DB_CONFIG` dictionary in `app.py`.

## Features

### Data Loading
- Automatically detects available tables in the database
- Performs intelligent column detection for dates, categories, and numeric values
- Includes comprehensive data cleaning (missing values, duplicates, data types)

### Visualizations
- **Line Chart**: Shows trends over time with automatic date column detection
- **Bar Chart**: Displays categorical comparisons with top 10 categories
- **Scatter Plot**: Analyzes relationships between numeric variables with optional color coding
- **Pie Chart**: Shows percentage distribution across categories
- **Histogram**: Displays data distribution with customizable bins
- **Box Plot**: Provides summary statistics and outlier detection
- **Heatmap**: Shows correlation matrix between numeric variables
- **Bubble Chart**: Scatter plot with bubble size based on a third variable

### Interactive Controls
- Category dropdown filter
- Date range picker
- Age range slider for numeric filtering
- Refresh button to reload data
- Real-time chart updates based on filter selections
- Interactive data table with pagination

### Error Handling
- Database connection failures gracefully fall back to sample data
- Comprehensive logging for debugging
- Input validation and error recovery

## Sample Data

If the database connection fails, the application will automatically generate sample insurance data including:
- Date ranges
- Insurance categories (Health, Auto, Life, Property)
- Amount, premium, and claims data

## Customization

The dashboard is designed to be flexible and will automatically adapt to your database schema. It will:
- Detect date columns for time-series analysis
- Identify categorical columns for grouping
- Find numeric columns for calculations and visualizations
- Handle various data types and formats

## Troubleshooting

1. **Database Connection Issues**: Check that MySQL is running and credentials are correct
2. **Missing Data**: Ensure your database has data in the expected tables
3. **Port Conflicts**: Change the port in the `app.run_server()` call if 8050 is in use

## Dependencies

- dash==2.14.2
- plotly==5.17.0
- pandas==2.1.3
- mysql-connector-python==8.2.0
- dash-bootstrap-components==1.5.0
- python-dotenv==1.0.0
