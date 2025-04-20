# config.py
# Parameters and constants for the balance_predictions project

DATABASE_URI = "your_database_uri_here"
API_ENDPOINT = "your_api_endpoint_here"

# Feature engineering parameters
FEATURE_WINDOW_SIZE = 30

# Model training parameters
CV_FOLDS = 5
HYPERPARAMETER_SEARCH_SPACE = {
    # example hyperparameters
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
}

# Drift detection parameters
DRIFT_DETECTION_WINDOW = 100
