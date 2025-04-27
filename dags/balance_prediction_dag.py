from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
import warnings
from config.config_loader import CONFIG
from src.prod.operators.data_collection import data_collection_task
from src.prod.operators.model_training import model_training_task
from src.prod.operators.scoring import scoring_task
from src.prod.operators.drift_detection import drift_detection_task
from src.prod.operators.feature_selection import feature_selection_task

import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)



def check_weekly_retraining(**context):
    """Check if we need to retrain based on weekly schedule"""
    execution_date = context['dag_run'].conf.get('date', '2021-01-01')
    execution_date = datetime.strptime(execution_date, '%Y-%m-%d')
    is_weekend = execution_date.weekday() >= 6  # 5 is Saturday, 6 is Sunday
    
    logger.info(f"Checking weekly retraining for date {execution_date}")
    logger.info(f"Is weekend: {is_weekend}")
    
    return is_weekend

def branch_on_drift_or_weekly(**context):
    """Branch based on drift detection or weekly retraining"""
    execution_date = context.get('execution_date', datetime.now())
    logger.info(f"Branching decision for date {execution_date}")
    
    # Get drift detection result from XCom
    drift_detected = context['ti'].xcom_pull(task_ids='detect_drift')
    weekly_retraining = check_weekly_retraining(**context)
    
    logger.info(f"Drift detected: {drift_detected}")
    logger.info(f"Weekly retraining: {weekly_retraining}")
    
    if drift_detected or weekly_retraining:
        return 'feature_selection'
    return 'to_scoring'

def make_next(context, dag_run_obj):
    """Функция-генератор для TriggerDagRunOperator"""
    # прочитаем дату из текущего conf
    last_date = context['dag_run'].conf.get('date')
    if not last_date:
        # если первый запуск — можно взять execution_date
        last_date = datetime(2021, 1, 1)
    # увеличим на день
    next_dt = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # остановка по условию (например, пока не дойдём до today)
    if next_dt <= datetime.today().strftime('%Y-%m-%d'):
        dag_run_obj.payload = {'date': next_dt}
        return dag_run_obj
    # иначе не триггерим
    return last_date

with DAG(
    'balance_prediction',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Balance prediction pipeline',
    schedule_interval='0 0 * * *',  # Daily at midnight
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2025, 1, 1),
    is_paused_upon_creation=True,
    tags=['balance', 'prediction'],
) as dag:
    
    # Data collection task
    collect_data = PythonOperator(
        task_id='collect_data',
        python_callable=data_collection_task,
        provide_context=True,
    )
    
    # Drift detection task
    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=drift_detection_task,
        provide_context=True,
    )
    
    # Branching task
    branch = BranchPythonOperator(
        task_id='branch',
        python_callable=branch_on_drift_or_weekly,
        provide_context=True,
    )
    
    # Model retraining task

    feature_selection = PythonOperator(
        task_id='feature_selection',
        python_callable=feature_selection_task,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # вместо all_success
    )
    retrain_model = PythonOperator(
        task_id='retrain_model',
        python_callable=model_training_task,
        provide_context=True,
    )
    to_scoring = DummyOperator(
        task_id='to_scoring',
        trigger_rule=TriggerRule.ONE_SUCCESS
    )
    # Scoring task
    score_data = PythonOperator(
        task_id='score_data',
        python_callable=scoring_task,
        provide_context=True,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # вместо all_success
    )

    # И в зависимостих можно оставить то, что у вас:
    collect_data >> detect_drift >> branch
    branch >> [feature_selection, to_scoring]
    feature_selection >> retrain_model
    retrain_model >> score_data
    to_scoring >> score_data