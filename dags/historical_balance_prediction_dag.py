from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import logging
from datetime import timezone

logger = logging.getLogger(__name__)

# Устанавливаем дату начала
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)

def get_next_date(**context):
    """Calculate next date to process and stop if we've reached current date"""
    # При первом запуске используем START_DATE
    if context['ti'].try_number == 1:
        execution_date = START_DATE
    else:
        execution_date = context.get('execution_date', START_DATE)
    
    next_date = execution_date + timedelta(days=1)
    
    logger.info("=== Date Information ===")
    logger.info(f"Current execution_date: {execution_date}")
    logger.info(f"Current execution_date type: {type(execution_date)}")
    logger.info(f"Current execution_date tzinfo: {execution_date.tzinfo}")
    logger.info(f"Next date: {next_date}")
    logger.info(f"Next date type: {type(next_date)}")
    logger.info(f"Next date tzinfo: {next_date.tzinfo}")
    logger.info(f"Current time: {datetime.now(timezone.utc)}")
    logger.info(f"Try number: {context['ti'].try_number}")
    logger.info("=====================")
    
    if next_date > datetime.now(timezone.utc):
        logger.info("Reached current date, stopping historical processing")
        return None
    
    return next_date

def trigger_next_run(**context):
    """Trigger next run of the DAG with the next date"""
    next_date = context['ti'].xcom_pull(task_ids='get_next_date')
    
    if next_date is None:
        logger.info("No more dates to process")
        return
    
    logger.info("=== Trigger Information ===")
    logger.info(f"Next date from XCom: {next_date}")
    logger.info(f"Next date type: {type(next_date)}")
    logger.info(f"Next date tzinfo: {next_date.tzinfo}")
    logger.info("=======================")
    
    # Create a new DAG run
    from airflow.models import DagRun
    from airflow.utils.state import DagRunState
    
    dag = context['dag']
    run_id = f"manual__{next_date.strftime('%Y%m%dT%H%M%S')}"
    
    # Check if run already exists
    existing_runs = DagRun.find(
        dag_id=dag.dag_id,
        execution_date=next_date
    )
    
    if existing_runs:
        logger.info(f"Found existing runs for date {next_date}:")
        for run in existing_runs:
            logger.info(f"Run ID: {run.run_id}, State: {run.state}")
        
        # Check if any run is in running state
        running_runs = [run for run in existing_runs if run.state == DagRunState.RUNNING]
        if running_runs:
            logger.info(f"Run already in progress for date {next_date}")
            return
    
    # Create new run with explicit execution date
    try:
        dag.create_dagrun(
            run_id=run_id,
            execution_date=next_date,
            state=DagRunState.RUNNING,
            conf={
                'execution_date': next_date.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'trigger_logical_date': next_date.strftime('%Y-%m-%dT%H:%M:%S%z')
            },
            external_trigger=True
        )
        logger.info(f"Successfully created new DAG run for date {next_date}")
    except Exception as e:
        logger.error(f"Error creating DAG run: {str(e)}")
        raise

with DAG(
    'historical_balance_prediction',
    default_args={
        'owner': 'airflow',
        'depends_on_past': True,
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Historical balance prediction pipeline',
    schedule_interval='@once',
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    tags=['balance', 'prediction', 'historical'],
) as dag:
    
    # Trigger main DAG with current execution date
    trigger_main_dag = TriggerDagRunOperator(
        task_id='trigger_main_dag',
        trigger_dag_id='balance_prediction',
        conf={
            'execution_date': '{{ execution_date }}',
            'trigger_logical_date': '{{ execution_date }}'
        },
        wait_for_completion=True,
    )
    
    # Get next date
    get_next_date_task = PythonOperator(
        task_id='get_next_date',
        python_callable=get_next_date,
        provide_context=True,
    )
    
    # Trigger next run
    trigger_next_run_task = PythonOperator(
        task_id='trigger_next_run',
        python_callable=trigger_next_run,
        provide_context=True,
    )
    
    # Define task dependencies
    trigger_main_dag >> get_next_date_task >> trigger_next_run_task 