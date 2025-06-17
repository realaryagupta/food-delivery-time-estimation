import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import json
from pathlib import Path
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, 
                                 OrdinalEncoder, PowerTransformer)
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
np.random.seed(42)
plt.style.use('ggplot')

# Constants
FEATURE_CATEGORIES = {
    'numerical': ["ratings", "distance"],
    'nominal': ['is_festival', 'is_weekend', 'day_of_week'],
    'ordinal': [
        "distance_type",
        "weather", 
        "order_type",
        "city_category",
        "order_time_of_day"
    ],
    'target': 'time'
}

ORDINAL_MAPPINGS = {
    "distance_type": ["short", "medium", "long", "very_long"],
    "weather": ['sunny', 'cloudy', 'windy', 'fog', 'stormy', 'sandstorms'],
    "order_type": ['snack', 'drinks', 'buffet', 'meal'],
    "city_category": ['urban', 'metropolitian', 'semi-urban'],
    "order_time_of_day": ['morning', 'afternoon', 'evening', 'night']
}

PATHS = {
    'data': '../data/processed/final_data.csv',
    'models': '../models',
    'model_file': 'stacking_regressor_model.joblib',
    'metadata': 'stacking_regressor_metadata.json',
    'performance_plot': 'stacking_regressor_performance.png',
    'feature_importance': 'feature_importance.png'
}

class DeliveryTimePredictor:
    """Main class for the delivery time prediction pipeline"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_importances = None
        
    def _ensure_directory_exists(self, path):
        """Ensure directory exists, create if it doesn't"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def load_data(self, data_path):
        """Load and validate the dataset"""
        print("Loading and preparing data...")
        data = pd.read_csv(data_path)
        
        # Data validation
        self._validate_data(data)
        
        # Clean data
        data_clean = data.dropna()
        print(f"\nDataset cleaned: {data_clean.shape[0]} records remaining")
        
        return data_clean
    
    def _validate_data(self, data):
        """Validate the input dataset"""
        print(f"\nRaw dataset shape: {data.shape}")
        print(f"Missing values:\n{data.isnull().sum()}")
        
        # Check required columns
        required_cols = (FEATURE_CATEGORIES['numerical'] + 
                        FEATURE_CATEGORIES['nominal'] + 
                        FEATURE_CATEGORIES['ordinal'] + 
                        [FEATURE_CATEGORIES['target']])
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check numerical features
        for num_col in FEATURE_CATEGORIES['numerical']:
            if not pd.api.types.is_numeric_dtype(data[num_col]):
                raise ValueError(f"Numerical column {num_col} contains non-numeric values")
    
    def create_preprocessor(self):
        """Create feature preprocessing pipeline"""
        print("\nCreating preprocessing pipeline...")
        
        transformers = [
            ('num', MinMaxScaler(), FEATURE_CATEGORIES['numerical']),
            ('nom_cat', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False,
                drop='if_binary'
            ), FEATURE_CATEGORIES['nominal']),
            ('ord_cat', OrdinalEncoder(
                categories=[ORDINAL_MAPPINGS[col] for col in FEATURE_CATEGORIES['ordinal']],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ), FEATURE_CATEGORIES['ordinal'])
        ]
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def create_model(self):
        """Create the stacking ensemble model"""
        print("\nCreating stacking ensemble model...")
        
        # Base models
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        lgbm_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        # Pipelines
        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(**rf_params))
        ])
        
        lgbm_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', lgb.LGBMRegressor(**lgbm_params))
        ])
        
        # Stacking model
        stacking_regressor = StackingRegressor(
            estimators=[
                ('random_forest', rf_pipeline),
                ('lightgbm', lgbm_pipeline)
            ],
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1
        )
        
        # Target transformation
        model = TransformedTargetRegressor(
            regressor=stacking_regressor,
            transformer=PowerTransformer(method='yeo-johnson', standardize=True)
        )
        
        return model
    
    def evaluate_model(self, X, y, cv=5):
        """Evaluate model performance using cross-validation"""
        print("\nEvaluating model performance...")
        
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = (-cv_scores).mean()
        
        print(f"Cross-validated RMSE: {cv_rmse:.4f}")
        return cv_rmse
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        # Store feature importances
        self._calculate_feature_importances(X_train, y_train)
    
    def _calculate_feature_importances(self, X, y):
        """Calculate and store feature importances"""
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', rf_model)
        ])
        rf_pipeline.fit(X, y)
        
        # Get feature names
        feature_names = self._get_feature_names()
        self.feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_pipeline.named_steps['regressor'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    def _get_feature_names(self):
        """Get feature names after preprocessing"""
        feature_names = []
        
        # Numerical features
        feature_names.extend(FEATURE_CATEGORIES['numerical'])
        
        # Nominal categorical features
        ohe = self.preprocessor.named_transformers_['nom_cat']
        ohe_features = ohe.get_feature_names_out(FEATURE_CATEGORIES['nominal'])
        feature_names.extend(ohe_features)
        
        # Ordinal categorical features
        feature_names.extend(FEATURE_CATEGORIES['ordinal'])
        
        return feature_names
    
    def save_model(self):
        """Save model and metadata"""
        print("\nSaving model artifacts...")
        
        # Ensure directory exists
        model_path = os.path.join(PATHS['models'], PATHS['model_file'])
        self._ensure_directory_exists(model_path)
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'feature_categories': FEATURE_CATEGORIES,
            'ordinal_mappings': ORDINAL_MAPPINGS,
            'feature_importances': self.feature_importances.to_dict()
        }
        
        metadata_path = os.path.join(PATHS['models'], PATHS['metadata'])
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to: {metadata_path}")
    
    def plot_results(self, X_test, y_test):
        """Create evaluation visualizations"""
        print("\nGenerating performance visualizations...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Set style for all plots
        plt.style.use('ggplot')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='steelblue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[0, 0].set_title('Actual vs Predicted Delivery Time', fontsize=12)
        axes[0, 0].set_xlabel('Actual', fontsize=10)
        axes[0, 0].set_ylabel('Predicted', fontsize=10)
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 2. Residuals plot
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='steelblue')
        axes[0, 1].axhline(0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals Plot', fontsize=12)
        axes[0, 1].set_xlabel('Predicted', fontsize=10)
        axes[0, 1].set_ylabel('Residuals', fontsize=10)
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)
        
        # 3. Feature importance
        top_features = self.feature_importances.head(15)
        axes[1, 0].barh(top_features['feature'], top_features['importance'], color='steelblue')
        axes[1, 0].set_title('Top 15 Feature Importances', fontsize=12)
        axes[1, 0].set_xlabel('Importance Score', fontsize=10)
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='steelblue')
        axes[1, 1].axvline(0, color='r', linestyle='--')
        axes[1, 1].set_title('Error Distribution', fontsize=12)
        axes[1, 1].set_xlabel('Prediction Error', fontsize=10)
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(PATHS['models'], PATHS['performance_plot'])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plots saved to: {plot_path}")

def main():
    try:
        # Initialize pipeline
        predictor = DeliveryTimePredictor()
        
        # Load data
        data = predictor.load_data(PATHS['data'])
        
        # Split data
        X = data.drop(FEATURE_CATEGORIES['target'], axis=1)
        y = data[FEATURE_CATEGORIES['target']]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        predictor.preprocessor = predictor.create_preprocessor()
        predictor.model = predictor.create_model()
        predictor.train(X_train, y_train)
        
        # Evaluate
        cv_score = predictor.evaluate_model(X_train, y_train)
        y_pred = predictor.model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        print("\nModel Evaluation Summary:")
        print(f"- Cross-validated RMSE: {cv_score:.4f}")
        print(f"- Test RMSE: {test_rmse:.4f}")
        print(f"- Test R²: {test_r2:.4f}")
        
        # Save and visualize
        predictor.save_model()
        predictor.plot_results(X_test, y_test)
        
        print("\nPipeline executed successfully!")
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()