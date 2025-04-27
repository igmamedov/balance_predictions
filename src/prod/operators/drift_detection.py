import pandas as pd
from datetime import datetime
import logging
import boto3
from src.utils import read_from_s3
import os
from config.config_loader import CONFIG
from src.drift_detector import ChangePoint


logger = logging.getLogger(__name__)


def drift_detection_task(**context):
    """
    Task to detect drift in the data.
    Returns 1 if drift is detected, 0 otherwise.
    """
    session = boto3.session.Session()
    s3_client = session.client(
            service_name='s3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
    
    try:
        # Get current date
        dag_run = context['dag_run']
        date = dag_run.conf.get('date') if dag_run and dag_run.conf else '2021-01-01'
        logger.info(f'Дата скоринга: {date}')
        date = datetime.strptime(date, '%Y-%m-%d')
        current_date = date.date()

        # Load current data
        current_data = read_from_s3(
            s3_client=s3_client,
            bucket=os.environ['AWS_BUCKET'],
            key=CONFIG['s3']['target_path']
        )
        
        # Log target dataset information
        logger.info("Target dataset information:")
        logger.info(f"Columns: {list(current_data.columns)}")
        logger.info(f"Shape: {current_data.shape}")
        logger.info(f"Data types:\n{current_data.dtypes}")
        logger.info(f"Sample data:\n{current_data.head()}")
        
        target_date_col = CONFIG['features']['lags']['target']['date_col']
        current_data[target_date_col] = pd.to_datetime(current_data[target_date_col])
        current_data.set_index(target_date_col, inplace=True)
        current_data = current_data['Balance']

      
        # Get last available date
        last_date = current_date
        logger.info(f"Last available date in dataset: {last_date}")
        
        # Get value for last date
        last_value = current_data[current_data.index.date == last_date].iloc[0]
        logger.info(f"Value for last date {last_date}: {last_value}")

        cp = ChangePoint(current_data)
        flag, _ = cp.calc_statistics(
            last_value,
            last_date
        )
        
        # Log drift detection result
        logger.info(f"Drift detection result: {'Detected' if flag else 'Not detected'}")
        
        # Push drift flag to XCom
        context['ti'].xcom_push(key='drift_detected', value=flag)
        
        # Also return the value for backward compatibility
        return flag
        
    except Exception as e:
        logger.error(f"Error in drift detection task: {str(e)}")
        raise