import pandas as pd
import boto3
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config.config_loader import CONFIG
from src.utils import init_mlflow, get_artifacts_by_run_name, log_experiment

def detect_drift(model, data, selected_features, scaler):
    """Detect data drift"""
    X = data[selected_features]
    y = data[CONFIG['model']['target_column']]
    
    # Remove rows with missing values
    X = X.dropna()
    y = y[X.index]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Check for drift
    is_drift = mae > CONFIG['model']['max_mae']
    
    return is_drift, mae, mse

def make_prediction(model, data, selected_features, scaler):
    """Make prediction for the next period"""
    # Get the latest data point
    last_data = data.iloc[-1][selected_features].values.reshape(1, -1)
    last_data_scaled = scaler.transform(last_data)
    
    # Make prediction
    prediction = model.predict(last_data_scaled)[0]
    
    return prediction

def scoring_task():
    """Task for scoring and drift detection"""
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
    
    # Get latest model
    artifacts = get_artifacts_by_run_name(CONFIG['mlflow']['model_name'], CONFIG['mlflow']['experiment_name'])
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]
    
    # Detect drift
    is_drift, mae, mse = detect_drift(model, data, selected_features, scaler)
    
    if not is_drift:
        # Make prediction
        prediction = make_prediction(model, data, selected_features, scaler)
        
        # Log prediction
        log_experiment(
            model=model,
            model_name=f"{CONFIG['mlflow']['model_name']}_prediction",
            params={
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "is_drift": is_drift
            },
            metrics={
                "prediction": prediction,
                "mae": mae,
                "mse": mse
            }
        )
        
        return prediction
    else:
        return None 