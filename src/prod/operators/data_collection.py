import pandas as pd
import boto3
from datetime import datetime
from config.config_loader import CONFIG
from src.feature_engineering import FeatureEngineering
from src.utils import read_from_s3, write_to_s3
import logging
import os
from airflow.hooks.base import BaseHook

logger = logging.getLogger(__name__)


def _extract_config():
    """Extract and validate configuration parameters"""
    s3_config = CONFIG['s3']
    features_config = CONFIG['features']
    
    return {
        's3_bucket': s3_config['bucket'],
        'data_path': s3_config['data_path'],
        'work_data': {
            'path': features_config['lags']['work']['path'],
            'date_col': features_config['lags']['work']['date_col']
        },
        'target_data': {
            'path': features_config['lags']['target']['path'],
            'date_col': features_config['lags']['target']['date_col']
        },
        'lags_params': features_config['lags_params'],
        'external_features': features_config['external_feats']
    }

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

def _process_external_features(s3_client, bucket, features_config, current_date):
    """Process all external features"""
    features_df = pd.DataFrame()
    
    for feat_name, feat_config in features_config.items():
        try:
            logger.info(f"Processing {feat_name}...")
            ext_data = _load_and_process_data(
                s3_client=s3_client,
                bucket=bucket,
                key=feat_config['path'],
                date_col=feat_config['date_col']
            )
            ext_data = ext_data[ext_data.index.date <= current_date]
            
            if features_df.empty:
                features_df = ext_data
            else:
                features_df = features_df.join(ext_data, how='left')
                
        except Exception as e:
            logger.error(f"Error processing external feature {feat_name}: {str(e)}")
            continue
    
    return features_df

def load_new_data():
    """Load and process all available data from S3"""
    try:
        # Extract configuration
        config = _extract_config()
        session = boto3.session.Session()
        s3_client = session.client(
            service_name='s3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
        )
        current_date = datetime.now().date()
        
        # 1. Load and process lag features
        logger.info("Loading lag features...")
        
        # Load work data
        work_data = _load_and_process_data(
            s3_client=s3_client,
            bucket=config['s3_bucket'],
            key=config['work_data']['path'],
            date_col=config['work_data']['date_col']
        )
        logger.info(f"Work data shape: {work_data.shape}")
        
        # Load target data
        target_data = _load_and_process_data(
            s3_client=s3_client,
            bucket=config['s3_bucket'],
            key=config['target_data']['path'],
            date_col=config['target_data']['date_col']
        )
        logger.info(f"Target data shape: {target_data.shape}")
        
        # 2. Process target data
        logger.info("Processing target data...")
        target_data = target_data[config['lags_params']['attrs']]
        target_data = target_data.shift(1)
        logger.info(f"Processed target data shape: {target_data.shape}")
        
        # 3. Generate features
        logger.info("Generating features...")
        feature_engineer = FeatureEngineering(
            df=target_data,
            calendar_df=work_data,
            n_lags=config['lags_params']['n_lags'],
            attrs=config['lags_params']['attrs'],
            tax_days=config['lags_params']['tax_days'],
            alphas=config['lags_params']['alphas']
        )
        features_df = feature_engineer.generate_features()
        logger.info(f"Generated features shape: {features_df.shape}")
        
        # 4. Process external features
        logger.info("Loading external features...")
        external_features = _process_external_features(
            s3_client=s3_client,
            bucket=config['s3_bucket'],
            features_config=config['external_features'],
            current_date=current_date
        )
        logger.info(f"External features shape: {external_features.shape}")
        
        # Join all features
        if not external_features.empty:
            features_df = features_df.join(external_features, how='left')
            logger.info(f"Final features shape after joining external features: {features_df.shape}")
        
        # 5. Save final data
        logger.info("Saving final data...")
        
        # Delete existing data if present
        try:
            s3_client.delete_object(
                Bucket=config['s3_bucket'],
                Key=config['data_path']
            )
            logger.info("Deleted existing data file")
        except Exception as e:
            logger.warning(f"No existing data file to delete: {str(e)}")
        
        # Save new data
        features_df = features_df.reset_index()
        write_to_s3(
            s3_client=s3_client,
            df=features_df,
            bucket=config['s3_bucket'],
            key=config['data_path'],
            index=False
        )
        
        logger.info(f"Successfully processed and saved all data. Final shape: {features_df.shape}")
        return features_df
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise

def data_collection_task():
    """Task for data collection and feature engineering"""
    try:
        return load_new_data()
    except Exception as e:
        logger.error(f"Data collection task failed: {str(e)}")
        raise 