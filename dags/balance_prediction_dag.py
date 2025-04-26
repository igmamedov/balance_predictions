from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
import warnings
from config.config_loader import CONFIG
from src.prod.operators.data_collection import data_collection_task
from src.prod.operators.model_training import model_training_task
from src.prod.operators.scoring import scoring_task
warnings.filterwarnings('ignore')


with DAG(
    'balance_prediction',
    default_args=CONFIG['airflow'],
    description='Balance prediction pipeline',
    schedule_interval=CONFIG['airflow']['schedule_interval'],
    start_date=datetime.strptime(CONFIG['airflow']['start_date'], "%Y-%m-%d"),
    catchup=False,
) as dag:
    
    # Print MLflow environment variables
    print_mlflow_env = BashOperator(
        task_id='print_mlflow_env',
        bash_command="""
            echo "MLflow Environment Variables:"
            echo "MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
            echo "MLFLOW_S3_ENDPOINT_URL: ${MLFLOW_S3_ENDPOINT_URL}"
            echo "MLFLOW_S3_BUCKET: ${MLFLOW_S3_BUCKET}"
            echo "MLFLOW_ARTIFACT_ROOT: ${MLFLOW_ARTIFACT_ROOT}"
            echo "AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION}"
            echo "AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}"
        """,
    )
    
    # Data collection task
    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=data_collection_task,
    )
    
    # Model training task
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=model_training_task,
    )
    
    # Scoring task
    make_prediction = PythonOperator(
        task_id='make_prediction',
        python_callable=scoring_task,
    )
    
    # Define task dependencies
    print_mlflow_env  >> collect_data >> train_model >> make_prediction 