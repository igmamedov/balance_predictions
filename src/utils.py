# utils.py
# Helper functions

import os
import mlflow
import mlflow.sklearn
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

def init_mlflow(
    tracking_uri: str = "http://localhost:5001",
    experiment_name: str = "default_experiment",
    s3_endpoint_url: str = "https://storage.yandexcloud.net",
    s3_bucket: str = "balance-predictions"
) -> None:
    """
    Инициализация подключения к MLflow
    
    Args:
        tracking_uri: URI MLflow сервера
        experiment_name: Название эксперимента
        s3_endpoint_url: URL S3 хранилища
        s3_bucket: Название S3 бакета
    """
    # Установка URI для MLflow
    mlflow.set_tracking_uri(tracking_uri)
    
    # Настройка S3
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
    os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
    os.environ['MLFLOW_S3_BUCKET'] = s3_bucket
    os.environ['MLFLOW_ARTIFACT_ROOT'] = f's3://{s3_bucket}/mlflow'
    
    # Создание эксперимента, если не существует
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

def log_experiment(
    model: Any,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None
) -> str:
    """
    Логирование эксперимента в MLflow
    
    Args:
        model: Обученная модель
        model_name: Название модели
        params: Параметры модели
        metrics: Метрики модели
        artifacts: Дополнительные артефакты
        run_name: Название запуска
        
    Returns:
        str: ID запуска
    """
    if run_name is None:
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        # Логирование параметров
        mlflow.log_params(params)
        
        # Логирование метрик
        mlflow.log_metrics(metrics)
        
        # Логирование модели
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )
        
        # Логирование дополнительных артефактов
        if artifacts:
            for name, artifact in artifacts.items():
                if isinstance(artifact, (pd.DataFrame, np.ndarray)):
                    mlflow.log_dict(artifact.to_dict() if isinstance(artifact, pd.DataFrame) else artifact.tolist(), 
                                  f"{name}.json")
                else:
                    mlflow.log_dict(artifact, f"{name}.json")
        
        return run.info.run_id

def get_experiment_artifacts(run_id: str) -> Dict[str, Any]:
    """
    Получение артефактов эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, Any]: Словарь с артефактами
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = {}
    
    # Получение информации о запуске
    run = client.get_run(run_id)
    
    # Получение артефактов
    for artifact in client.list_artifacts(run_id):
        if artifact.path.endswith('.json'):
            artifacts[artifact.path] = client.download_artifacts(run_id, artifact.path)
    
    # Получение модели
    model_path = f"runs:/{run_id}/model"
    artifacts['model'] = mlflow.sklearn.load_model(model_path)
    
    return artifacts

def get_experiment_metrics(run_id: str) -> Dict[str, float]:
    """
    Получение метрик эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, float]: Словарь с метриками
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.metrics

def get_experiment_params(run_id: str) -> Dict[str, str]:
    """
    Получение параметров эксперимента по ID запуска
    
    Args:
        run_id: ID запуска
        
    Returns:
        Dict[str, str]: Словарь с параметрами
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.params

def some_helper_function():
    pass
