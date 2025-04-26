from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
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
    collect_data >> train_model >> make_prediction 