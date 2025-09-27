"""
Machine Learning Insights for Insurance Analytics
===============================================
"""

import mysql.connector
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, classification_report, r2_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceMLInsights:
    def __init__(self, db_config):
        self.db_config = db_config
        self.reg_model = None
        self.clf_model = None
        self.encoder = None
        self.feature_columns = None
        self.data = None
        
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
            logger.info("Connected to MySQL database")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def load_data(self):
        """Load data from database or create sample data"""
        try:
            if self.connect_database():
                # Try to get data from database
                self.cursor.execute("SELECT * FROM insurance;")
                rows = self.cursor.fetchall()
                if rows:
                    self.data = pd.DataFrame(rows)
                    logger.info(f"Loaded {len(self.data)} records from database")
                else:
                    self.data = self.create_sample_data()
                    logger.info("No data in database, using sample data")
            else:
                self.data = self.create_sample_data()
                logger.info("Database connection failed, using sample data")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.data = self.create_sample_data()
            logger.info("Using sample data due to error")
    
    def create_sample_data(self):
        """Create comprehensive sample data for ML training"""
        np.random.seed(42)
        n_records = 2000
        
        # Generate realistic insurance data
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
        
        data = pd.DataFrame({
            'age': ages,
            'sex': np.random.choice(['male', 'female'], n_records),
            'bmi': bmi,
            'children': children,
            'smoker': ['yes' if s else 'no' for s in smoker],
            'charges': charges,
            'region': regions
        })
        
        return data
    
    def create_features(self):
        """Create advanced features for ML models"""
        logger.info("Creating advanced features...")
        
        # Interaction features
        self.data['age_bmi'] = self.data['age'] * self.data['bmi']
        self.data['children_smoker'] = self.data['children'] * self.data['smoker'].map({'yes':1,'no':0})
        
        # Polynomial features
        self.data['bmi_sq'] = self.data['bmi']**2
        self.data['age_sq'] = self.data['age']**2
        
        # Age bands
        self.data['age_band'] = pd.cut(self.data['age'], bins=[17,29,39,49,100], 
                                      labels=['18-29','30-39','40-49','50+'])
        
        # BMI bands
        self.data['bmi_band'] = pd.cut(self.data['bmi'], bins=[0,25,30,100], 
                                      labels=['normal','overweight','obese'])
        
        # Risk category
        self.data['risk_category'] = self.data.apply(
            lambda x: 'High' if x['smoker']=='yes' and x['bmi']>30 else 
                     ('Medium' if (25<=x['bmi']<=30) or x['age']>50 else 'Low'),
            axis=1
        )
        
        logger.info("Advanced features created successfully")
    
    def prepare_data(self):
        """Prepare data for ML models"""
        logger.info("Preparing data for ML models...")
        
        # Encode categorical variables
        categorical_cols = ['sex','smoker','region','age_band','bmi_band']
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded = self.encoder.fit_transform(self.data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_cols))
        
        # Combine with numerical data
        self.data_model = pd.concat([self.data.drop(columns=categorical_cols), encoded_df], axis=1)
        self.feature_columns = [col for col in self.data_model.columns if col not in ['charges', 'risk_category']]
        
        logger.info(f"Data prepared with {len(self.feature_columns)} features")
    
    def train_models(self):
        """Train ML models"""
        logger.info("Training ML models...")
        
        # Prepare features and targets
        X = self.data_model[self.feature_columns]
        y_charges = self.data_model['charges']
        y_risk = self.data_model['risk_category']
        
        # Split data
        X_train, X_test, y_train_charges, y_test_charges = train_test_split(
            X, y_charges, test_size=0.2, random_state=42
        )
        _, _, y_train_risk, y_test_risk = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        
        # Train Regression Model (Charges Prediction)
        logger.info("Training charges prediction model...")
        self.reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        self.reg_model.fit(X_train, y_train_charges)
        y_pred_charges = self.reg_model.predict(X_test)
        
        # Train Classification Model (Risk Prediction)
        logger.info("Training risk classification model...")
        self.clf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        self.clf_model.fit(X_train, y_train_risk)
        y_pred_risk = self.clf_model.predict(X_test)
        
        # Calculate metrics
        self.reg_metrics = {
            'mse': mean_squared_error(y_test_charges, y_pred_charges),
            'rmse': np.sqrt(mean_squared_error(y_test_charges, y_pred_charges)),
            'r2': r2_score(y_test_charges, y_pred_charges)
        }
        
        self.clf_metrics = classification_report(y_test_risk, y_pred_risk, output_dict=True)
        
        logger.info("Models trained successfully!")
        logger.info(f"Regression R²: {self.reg_metrics['r2']:.3f}")
        logger.info(f"Regression RMSE: ${self.reg_metrics['rmse']:,.0f}")
        
        # Save models
        joblib.dump(self.reg_model, 'charges_model.pkl')
        joblib.dump(self.clf_model, 'risk_model.pkl')
        joblib.dump(self.encoder, 'encoder.pkl')
        
        logger.info("Models saved successfully!")
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        if self.reg_model is None or self.clf_model is None:
            return None, None
        
        # Regression feature importance
        reg_importance = pd.Series(
            self.reg_model.feature_importances_, 
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        # Classification feature importance
        clf_importance = pd.Series(
            self.clf_model.feature_importances_, 
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        return reg_importance, clf_importance
    
    def create_ml_visualizations(self):
        """Create ML-specific visualizations"""
        if self.reg_model is None or self.clf_model is None:
            return {}
        
        reg_importance, clf_importance = self.get_feature_importance()
        
        # Feature importance for charges prediction
        fig_charges_importance = px.bar(
            x=reg_importance.head(10).values,
            y=reg_importance.head(10).index,
            orientation='h',
            title="🔍 Top 10 Features for Charges Prediction",
            labels={'x': 'Importance', 'y': 'Features'},
            color=reg_importance.head(10).values,
            color_continuous_scale='Viridis'
        )
        fig_charges_importance.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Feature importance for risk classification
        fig_risk_importance = px.bar(
            x=clf_importance.head(10).values,
            y=clf_importance.head(10).index,
            orientation='h',
            title="🎯 Top 10 Features for Risk Classification",
            labels={'x': 'Importance', 'y': 'Features'},
            color=clf_importance.head(10).values,
            color_continuous_scale='Plasma'
        )
        fig_risk_importance.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Risk category distribution
        risk_dist = self.data['risk_category'].value_counts()
        fig_risk_dist = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="📊 Risk Category Distribution",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_risk_dist.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Age vs BMI with risk categories
        fig_risk_scatter = px.scatter(
            self.data, 
            x='age', 
            y='bmi', 
            color='risk_category',
            size='charges',
            title="🎯 Age vs BMI with Risk Categories",
            hover_data=['smoker', 'region', 'charges'],
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        fig_risk_scatter.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return {
            'charges_importance': fig_charges_importance,
            'risk_importance': fig_risk_importance,
            'risk_distribution': fig_risk_dist,
            'risk_scatter': fig_risk_scatter
        }
    
    def get_insights_summary(self):
        """Get summary of ML insights"""
        if self.reg_model is None or self.clf_model is None:
            return {}
        
        reg_importance, clf_importance = self.get_feature_importance()
        
        insights = {
            'total_records': len(self.data),
            'regression_r2': self.reg_metrics['r2'],
            'regression_rmse': self.reg_metrics['rmse'],
            'top_charges_factor': reg_importance.index[0],
            'top_risk_factor': clf_importance.index[0],
            'risk_distribution': self.data['risk_category'].value_counts().to_dict(),
            'avg_charges_by_risk': self.data.groupby('risk_category')['charges'].mean().to_dict()
        }
        
        return insights
    
    def predict_charges(self, age, sex, bmi, children, smoker, region):
        """Predict charges for new customer"""
        if self.reg_model is None or self.encoder is None:
            return None
        
        # Create input data
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Create features
        input_data['age_bmi'] = input_data['age'] * input_data['bmi']
        input_data['children_smoker'] = input_data['children'] * input_data['smoker'].map({'yes':1,'no':0})
        input_data['bmi_sq'] = input_data['bmi']**2
        input_data['age_sq'] = input_data['age']**2
        
        # Age and BMI bands
        input_data['age_band'] = pd.cut(input_data['age'], bins=[17,29,39,49,100], 
                                       labels=['18-29','30-39','40-49','50+'])
        input_data['bmi_band'] = pd.cut(input_data['bmi'], bins=[0,25,30,100], 
                                       labels=['normal','overweight','obese'])
        
        # Encode categorical variables
        categorical_cols = ['sex','smoker','region','age_band','bmi_band']
        encoded = self.encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_cols))
        
        # Combine features
        input_features = pd.concat([input_data.drop(columns=categorical_cols), encoded_df], axis=1)
        input_features = input_features[self.feature_columns]
        
        # Predict
        prediction = self.reg_model.predict(input_features)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def predict_risk(self, age, sex, bmi, children, smoker, region):
        """Predict risk category for new customer"""
        if self.clf_model is None or self.encoder is None:
            return None
        
        # Create input data (same as charges prediction)
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Create features
        input_data['age_bmi'] = input_data['age'] * input_data['bmi']
        input_data['children_smoker'] = input_data['children'] * input_data['smoker'].map({'yes':1,'no':0})
        input_data['bmi_sq'] = input_data['bmi']**2
        input_data['age_sq'] = input_data['age']**2
        
        # Age and BMI bands
        input_data['age_band'] = pd.cut(input_data['age'], bins=[17,29,39,49,100], 
                                       labels=['18-29','30-39','40-49','50+'])
        input_data['bmi_band'] = pd.cut(input_data['bmi'], bins=[0,25,30,100], 
                                       labels=['normal','overweight','obese'])
        
        # Encode categorical variables
        categorical_cols = ['sex','smoker','region','age_band','bmi_band']
        encoded = self.encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(categorical_cols))
        
        # Combine features
        input_features = pd.concat([input_data.drop(columns=categorical_cols), encoded_df], axis=1)
        input_features = input_features[self.feature_columns]
        
        # Predict
        prediction = self.clf_model.predict(input_features)[0]
        probabilities = self.clf_model.predict_proba(input_features)[0]
        
        return {
            'risk_category': prediction,
            'probabilities': dict(zip(self.clf_model.classes_, probabilities))
        }
    
    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()
        logger.info("Database connection closed")

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': 'localhost',
        'user': 'Osama',
        'password': 'YourPassword123',
        'database': 'insurance_db'
    }
    
    # Initialize ML insights
    ml_insights = InsuranceMLInsights(db_config)
    
    # Load data and train models
    ml_insights.load_data()
    ml_insights.create_features()
    ml_insights.prepare_data()
    ml_insights.train_models()
    
    # Get insights
    insights = ml_insights.get_insights_summary()
    print("\n" + "="*50)
    print("MACHINE LEARNING INSIGHTS SUMMARY")
    print("="*50)
    print(f"Total Records: {insights['total_records']:,}")
    print(f"Regression R² Score: {insights['regression_r2']:.3f}")
    print(f"Regression RMSE: ${insights['regression_rmse']:,.0f}")
    print(f"Top Charges Factor: {insights['top_charges_factor']}")
    print(f"Top Risk Factor: {insights['top_risk_factor']}")
    print("\nRisk Distribution:")
    for risk, count in insights['risk_distribution'].items():
        print(f"  {risk}: {count:,} customers")
    print("\nAverage Charges by Risk:")
    for risk, avg_charges in insights['avg_charges_by_risk'].items():
        print(f"  {risk}: ${avg_charges:,.0f}")
    
    # Close connection
    ml_insights.close_connection()
    
    print("\nModels trained and saved successfully!")
    print("Files created: charges_model.pkl, risk_model.pkl, encoder.pkl")
