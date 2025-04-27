import pandas as pd
import boto3
from datetime import datetime, timedelta
from src.utils import write_to_s3
import os 
from src.feature_selection import FeatureSelection
from io import StringIO
import pandas as pd

import boto3
import pickle
import logging

logger = logging.getLogger(__name__)


def feature_selection_task(**context):
    """Task for model training and evaluation"""
    session = boto3.session.Session()
    s3_client = session.client(
            service_name='s3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    # Initialize MLflow
    # init_mlflow(
    #     tracking_uri='http://host.docker.internal:5001',
    #     experiment_name='RegressBoost',
    #     s3_bucket=os.environ['MLFLOW_S3_BUCKET']
    # )
    # client = mlflow.tracking.MlflowClient()

    dag_run = context['dag_run']
    date = dag_run.conf.get('date') if dag_run and dag_run.conf else '2021-01-01'
    logger.info(f'Дата скоринга: {date}')
    date = datetime.strptime(date, '%Y-%m-%d')
    last_date = date.date()
    test_border = date - timedelta(days=30)


    bytes_data = s3_client.get_object(Bucket='balance-predictions', Key="prod/balance_data.csv")['Body'].read()
    csv_data = StringIO(str(bytes_data,'utf-8')) 
    X = pd.read_csv(csv_data)
    X['Timestamp'] = pd.to_datetime(X['Timestamp'])
    X = X.set_index('Timestamp').bfill()

    bytes_data = s3_client.get_object(Bucket='balance-predictions', Key="target/flow.csv")['Body'].read()
    csv_data = StringIO(str(bytes_data,'utf-8')) 
    y = pd.read_csv(csv_data)
    y['Timestamp'] = pd.to_datetime(y['Timestamp'])
    y = y.set_index('Timestamp').bfill()['Balance']


    # Selection
    train_idx = len(X[X.index < test_border])

    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]

    fs = FeatureSelection(X_train, y_train)
    top_features = list(fs.select_best(n=50))
    # забираем топ фичи из прошлой таски
    logger.info(f'Отобранные фичи для модели: {top_features}')

    s3_client.put_object(
    Bucket='balance-predictions',
    Key='prod/top_features.pkl',
    Body= pickle.dumps(top_features),
    ContentType='application/octet-stream'
)
    logger.info('Успешная запись списка топ признаков')

