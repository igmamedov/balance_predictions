import pandas as pd
import numpy as np
import boto3
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from config.config_loader import CONFIG
from src.utils import init_mlflow, log_experiment

def select_features(data):
    """Feature selection"""
    X = data.drop([CONFIG['model']['target_column'], 'date'], axis=1)
    y = data[CONFIG['model']['target_column']]
    
    # Remove rows with missing values
    X = X.dropna()
    y = y[X.index]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Embedded method
    selector_embedded = SelectFromModel(
        RandomForestRegressor(
            n_estimators=CONFIG['feature_selection']['n_estimators'],
            random_state=CONFIG['feature_selection']['random_state']
        )
    )
    selector_embedded.fit(X_scaled, y)
    features_embedded = X.columns[selector_embedded.get_support()]
    
    # Wrapper method
    selector_wrapper = RFE(
        RandomForestRegressor(
            n_estimators=CONFIG['feature_selection']['n_estimators'],
            random_state=CONFIG['feature_selection']['random_state']
        ),
        n_features_to_select=CONFIG['feature_selection']['n_features_to_select']
    )
    selector_wrapper.fit(X_scaled, y)
    features_wrapper = X.columns[selector_wrapper.get_support()]
    
    # Filter method
    mi_scores = mutual_info_regression(X_scaled, y)
    features_filter = X.columns[mi_scores > np.mean(mi_scores)]
    
    # Combine features
    selected_features = list(set(features_embedded) | set(features_wrapper) | set(features_filter))
    
    return selected_features, scaler

def train_model(data, selected_features):
    """Train model with hyperparameter tuning"""
    X = data[selected_features]
    y = data[CONFIG['model']['target_column']]
    
    # Remove rows with missing values
    X = X.dropna()
    y = y[X.index]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=CONFIG['training']['n_splits'])
    
    # Grid search
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=CONFIG['feature_selection']['random_state']),
        CONFIG['training']['param_grid'],
        cv=tscv,
        scoring=CONFIG['training']['scoring'],
        n_jobs=CONFIG['training']['n_jobs']
    )
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    
    return best_model, best_params, best_score, scaler

def model_training_task():
    """Task for model training and evaluation"""
    # Initialize MLflow
    init_mlflow(
        tracking_uri=CONFIG['mlflow']['tracking_uri'],
        experiment_name=CONFIG['mlflow']['experiment_name'],
        s3_bucket=CONFIG['s3']['bucket']
    )
    
    # Load data
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=CONFIG['s3']['bucket'], Key=CONFIG['s3']['features_path'])
    data = pd.read_csv(obj['Body'])
    
    # Check if retraining is needed
    last_train_date = pd.to_datetime(data['date']).max() - timedelta(days=CONFIG['model']['retrain_days'])
    needs_retraining = pd.to_datetime(data['date']).max() > last_train_date
    
    if needs_retraining:
        # Select features
        selected_features, scaler = select_features(data)
        
        # Train model
        model, params, mae, scaler = train_model(data, selected_features)
        
        # Log experiment
        run_id = log_experiment(
            model=model,
            model_name=CONFIG['mlflow']['model_name'],
            params={
                "selected_features": selected_features,
                "retrain_days": CONFIG['model']['retrain_days'],
                "max_mae": CONFIG['model']['max_mae'],
                **params
            },
            metrics={"mae": mae},
            artifacts={
                "scaler": scaler,
                "selected_features": selected_features
            }
        )
        
        return run_id
    else:
        return None 