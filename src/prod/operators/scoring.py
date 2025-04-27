import pandas as pd
import boto3
import os 
import mlflow
from datetime import datetime
from src.utils import read_from_s3, write_to_s3
from config.config_loader import CONFIG
from src.utils import init_mlflow, get_artifacts_by_run_name, log_experiment
from src.utils import init_mlflow, get_experiment_artifacts, get_experiment_metrics, log_experiment

import logging

logger = logging.getLogger(__name__)

def _load_and_process_data(s3_client, bucket, key, date_col):
    """Load data from S3 and process dates"""
    try:
        data = read_from_s3(
            s3_client=s3_client,
            bucket=bucket,
            key=key
        )
        
        if data is None:
            raise ValueError(f"Failed to read data from S3: {bucket}/{key}")
            
        if date_col not in data.columns:
            raise ValueError(f"Column {date_col} not found in data from {bucket}/{key}")
            
        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace=True)
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from {bucket}/{key}: {str(e)}")
        raise


def scoring_task(**context):
    """Task for scoring and drift detection"""

    session = boto3.session.Session()
    s3_client = session.client(
            service_name='s3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    # Initialize MLflow
    init_mlflow(
        tracking_uri='http://host.docker.internal:5001',
        experiment_name='RegressBoost',
        s3_bucket=os.environ['MLFLOW_S3_BUCKET']
    )
    client = mlflow.tracking.MlflowClient()

    # Get scoring date
    dag_run = context['dag_run']
    date = dag_run.conf.get('date') if dag_run and dag_run.conf else '2021-01-01'
    logger.info(f'Дата скоринга: {date}')
    date = datetime.strptime(date, '%Y-%m-%d').date()

    # Get scoring features
    scoring_df = _load_and_process_data(
        s3_client,
        os.environ['MLFLOW_S3_BUCKET'],
        'prod/balance_data.csv',
        'Timestamp'
    )
    logger.info(f"Sample data:\n{scoring_df.head()}")
    on_date_data = scoring_df[scoring_df.index.date == date]

    
    experiment = client.get_experiment_by_name("RegressBoost")
    if not experiment:
        raise ValueError("Эксперимент 'RegressBoost' не найден")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("Запуски не найдены в эксперименте")
    
    latest_run = runs[0]
    run_id = latest_run.info.run_id
    logger.info(f'Полученный run_id модели: {run_id}')
    
    # Получение артефактов последнего запуска
    artifacts = get_experiment_artifacts(run_id)
    if not artifacts:
        raise ValueError("Артефакты не найдены для последнего запуска")
    
    model = artifacts.get("model")
    if model is None:
        raise ValueError("Модель не найдена в артефактах")
    # features = artifacts.get('features.json')
    on_date_data = on_date_data[model.feature_names_in_]
    logger.info(f'Данные для скоринга: {on_date_data.shape}')

    predicted_value = model.predict(on_date_data)
    logger.info(f'Предсказанное значение баланса: {predicted_value}')

    return predicted_value
