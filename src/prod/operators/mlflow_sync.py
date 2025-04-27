import os
import logging
import boto3
import mlflow
import requests
import json
from mlflow.tracking import MlflowClient
from typing import List, Tuple, Optional
import yaml
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_ui_state(mlflow_client: MlflowClient, expected_count: int) -> bool:
    """Verify that the UI reflects the expected state."""
    try:
        experiments = mlflow_client.search_experiments()
        actual_count = len(experiments)
        logger.info(f"UI shows {actual_count} experiments, expected {expected_count}")
        for exp in experiments:
            logger.info(f"Experiment: {exp.name} (ID: {exp.experiment_id})")
        return actual_count >= expected_count
    except Exception as e:
        logger.error(f"Error verifying UI state: {str(e)}")
        return False

def get_s3_client() -> boto3.client:
    """Initialize and return S3 client with configured credentials."""
    try:
        endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_DEFAULT_REGION')
        
        logger.info(f"Initializing S3 client with endpoint: {endpoint_url}")
        logger.info(f"Region: {region}")
        
        client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Test S3 connection
        bucket = os.getenv('MLFLOW_S3_BUCKET')
        client.head_bucket(Bucket=bucket)
        logger.info(f"Successfully connected to S3 bucket: {bucket}")
        
        return client
    except Exception as e:
        logger.error(f"Error initializing S3 client: {str(e)}")
        raise

def get_mlflow_client() -> MlflowClient:
    """Initialize and return MLflow client with configured tracking URI."""
    try:
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        logger.info(f"Initializing MLflow client with tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
        
        # Test MLflow connection
        client.list_experiments()
        logger.info("Successfully connected to MLflow server")
        
        return client
    except Exception as e:
        logger.error(f"Error initializing MLflow client: {str(e)}")
        raise

def get_experiment_runs(s3_client: boto3.client, bucket: str) -> List[Tuple[str, str, str]]:
    """
    Get all experiment runs from S3.
    Returns list of tuples (experiment_id, run_id, meta_path).
    """
    try:
        logger.info(f"Listing objects in bucket {bucket} with prefix 'mlflow/'")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='mlflow/')
        
        if 'Contents' not in response:
            logger.warning(f"No objects found in bucket {bucket}")
            return []
        
        runs = []
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('meta.yaml'):
                parts = key.split('/')
                if len(parts) >= 4:
                    experiment_id = parts[2]
                    run_id = parts[4]
                    runs.append((experiment_id, run_id, key))
                    logger.debug(f"Found run: experiment_id={experiment_id}, run_id={run_id}")
        
        logger.info(f"Found {len(runs)} experiment runs in S3")
        for exp_id, run_id, _ in runs:
            logger.info(f"Experiment ID: {exp_id}, Run ID: {run_id}")
        return runs
    except Exception as e:
        logger.error(f"Error getting experiment runs: {str(e)}")
        return []

def sync_experiment_to_mlflow(
    s3_client: boto3.client,
    mlflow_client: MlflowClient,
    bucket: str,
    experiment_id: str,
    run_id: str,
    meta_path: str
) -> bool:
    """
    Sync a single experiment run from S3 to MLflow.
    Returns True if successful, False otherwise.
    """
    try:
        logger.info(f"Syncing experiment {experiment_id}, run {run_id}")
        
        # Download metadata
        logger.info(f"Downloading metadata from {meta_path}")
        meta_response = s3_client.get_object(Bucket=bucket, Key=meta_path)
        meta_content = meta_response['Body'].read().decode('utf-8')
        meta_data = yaml.safe_load(meta_content)
        logger.info(f"Metadata content: {json.dumps(meta_data, indent=2)}")
        
        # Create or get experiment
        experiment = mlflow_client.get_experiment_by_name(experiment_id)
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_id}")
            experiment_id = mlflow_client.create_experiment(experiment_id)
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_id}")
        
        # Create run
        with mlflow.start_run(experiment_id=experiment_id, run_id=run_id) as run:
            logger.info(f"Created run with ID: {run.info.run_id}")
            
            # Log parameters
            if 'params' in meta_data:
                for key, value in meta_data['params'].items():
                    mlflow.log_param(key, value)
                    logger.info(f"Logged parameter: {key}={value}")
            
            # Log metrics
            if 'metrics' in meta_data:
                for key, value in meta_data['metrics'].items():
                    mlflow.log_metric(key, value)
                    logger.info(f"Logged metric: {key}={value}")
            
            # Log tags
            if 'tags' in meta_data:
                for key, value in meta_data['tags'].items():
                    mlflow.set_tag(key, value)
                    logger.info(f"Logged tag: {key}={value}")
            
            # Download and log artifacts
            artifact_prefix = f"mlflow/{experiment_id}/{run_id}/artifacts/"
            try:
                logger.info(f"Looking for artifacts in {artifact_prefix}")
                artifacts = s3_client.list_objects_v2(
                    Bucket=bucket,
                    Prefix=artifact_prefix
                )
                
                if 'Contents' in artifacts:
                    for artifact in artifacts['Contents']:
                        artifact_key = artifact['Key']
                        local_path = f"/tmp/{os.path.basename(artifact_key)}"
                        
                        logger.info(f"Downloading artifact: {artifact_key}")
                        s3_client.download_file(bucket, artifact_key, local_path)
                        mlflow.log_artifact(local_path)
                        logger.info(f"Logged artifact: {artifact_key}")
                        
                        # Clean up
                        os.remove(local_path)
                else:
                    logger.info("No artifacts found")
            except Exception as e:
                logger.warning(f"Error syncing artifacts: {str(e)}")
        
        # Verify UI state after sync
        if not verify_ui_state(mlflow_client, 1):
            logger.warning(f"UI state verification failed for experiment {experiment_id}")
            return False
        
        logger.info(f"Successfully synced experiment {experiment_id}, run {run_id}")
        return True
    except Exception as e:
        logger.error(f"Error syncing experiment {experiment_id}, run {run_id}: {str(e)}")
        return False

def sync_all_experiments() -> None:
    """Sync all experiments from S3 to MLflow."""
    try:
        logger.info("Starting sync process")
        
        s3_client = get_s3_client()
        mlflow_client = get_mlflow_client()
        bucket = os.getenv('MLFLOW_S3_BUCKET')
        
        if not bucket:
            raise ValueError("MLFLOW_S3_BUCKET environment variable is not set")
        
        # Get current experiment count
        initial_count = len(mlflow_client.search_experiments())
        logger.info(f"Initial experiment count: {initial_count}")
        
        # Get all experiment runs
        experiment_runs = get_experiment_runs(s3_client, bucket)
        
        if not experiment_runs:
            logger.warning("No experiment runs found to sync")
            return
        
        # Sync each experiment
        success_count = 0
        for experiment_id, run_id, meta_path in experiment_runs:
            if sync_experiment_to_mlflow(s3_client, mlflow_client, bucket, experiment_id, run_id, meta_path):
                success_count += 1
        
        # Final UI state verification
        final_count = len(mlflow_client.search_experiments())
        logger.info(f"Final experiment count: {final_count}")
        
        if final_count <= initial_count:
            logger.warning("No new experiments detected in UI after sync")
            logger.info("Checking MLflow server configuration...")
            logger.info(f"Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
            logger.info(f"Artifact Root: {os.getenv('MLFLOW_ARTIFACT_ROOT')}")
            logger.info(f"S3 Bucket: {bucket}")
        
        logger.info(f"Sync process completed. Successfully synced {success_count}/{len(experiment_runs)} runs")
    except Exception as e:
        logger.error(f"Error during sync process: {str(e)}")
        raise

if __name__ == "__main__":
    sync_all_experiments() 