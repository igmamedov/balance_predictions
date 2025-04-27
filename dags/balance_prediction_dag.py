from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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

def make_next(**context):
    """Функция-генератор для TriggerDagRunOperator"""
    # прочитаем дату из текущего conf
    dag_run = context['dag_run']
    last_date = dag_run.conf.get('date') if dag_run and dag_run.conf else '2021-01-01'
    
    # увеличим на день
    next_dt = datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)
    # остановка по условию (например, пока не дойдём до today)
    if next_dt > datetime(2021, 1, 15):
        next_dt = None
    else:
        next_dt = next_dt.strftime('%Y-%m-%d')
        logger.info(f'Next date: {next_dt}')
    context['ti'].xcom_push(key='next_date', value=next_dt)


def branch_by_xcom(**context):
    ti = context['ti']
    # таcк, который пушит в XCom ключ 'date'
    next_date = ti.xcom_pull(task_ids='calculate_next_date', key='next_date')
    logger.info(f'Next date: {next_date}')
    if next_date:
        return 'trigger_dag'
    else:
        return 'stop'

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
    trigger_next = PythonOperator(
        task_id='calculate_next_date',
        python_callable=make_next,
        provide_context=True    
    )

    branch_new_dag = BranchPythonOperator(
        task_id='branch_new_dag',
        python_callable=branch_by_xcom,
        provide_context=True,
    )

    trigger_dag = TriggerDagRunOperator(
        task_id='trigger_dag',
        trigger_dag_id='balance_prediction',
        conf={'date': '{{ ti.xcom_pull(task_ids="calculate_next_date", key="next_date") }}'},
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    stop = PythonOperator(
        task_id='stop',
        python_callable=lambda: print("No date found, stopping."),
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )
    # И в зависимостих можно оставить то, что у вас:
    collect_data >> detect_drift >> branch
    branch >> [feature_selection, to_scoring]
    feature_selection >> retrain_model
    retrain_model >> score_data
    to_scoring >> score_data >> trigger_next
    trigger_next >> branch_new_dag >> [trigger_dag, stop]