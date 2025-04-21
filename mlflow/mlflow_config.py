import mlflow
import os

def setup_mlflow():
    # Set MLflow tracking URI (e.g., local or remote server)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    # Set experiment name
    mlflow.set_experiment("balance_predictions_experiment")

    # Configure S3 artifact location
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "https://s3.amazonaws.com")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ["AWS_DEFAULT_REGION"] = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    print("MLflow configured with S3 artifact store and tracking URI.")

if __name__ == "__main__":
    setup_mlflow()
