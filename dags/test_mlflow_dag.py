from datetime import datetime, timedelta
import numpy as np
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from src.utils import init_mlflow, get_experiment_artifacts, get_experiment_metrics, log_experiment

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def test_model():
    """Тестирование модели из MLflow"""
    # Инициализация MLflow
    init_mlflow(
        experiment_name="test_experiment_utils",
        s3_bucket="balance-predictions"
    )
    
    # Получение последней модели из эксперимента
    artifacts = get_experiment_artifacts("latest")
    if not artifacts:
        raise ValueError("Модель не найдена в MLflow")
    
    model = artifacts.get("model")
    if model is None:
        raise ValueError("Модель не найдена в артефактах")
    
    # Создание тестовых данных
    X_test = np.array([[6], [7], [8], [9], [10]])
    
    # Выполнение предсказаний
    y_pred = model.predict(X_test)
    
    # Получение метрик оригинальной модели
    metrics = get_experiment_metrics("latest")
    
    # Логирование результатов теста
    run_id = log_experiment(
        model=model,
        model_name="test_model",
        params={
            "model_type": "LinearRegression",
            "test_size": len(X_test)
        },
        metrics={
            "original_mse": metrics.get("mse", 0),
            "test_samples": len(X_test)
        },
        artifacts={
            "X_test": X_test,
            "y_pred": y_pred
        }
    )
    
    print(f"Тестирование завершено. Run ID: {run_id}")
    print(f"Предсказания: {y_pred}")
    return run_id

with DAG(
    'test_mlflow_model',
    default_args=default_args,
    description='Тестирование модели из MLflow',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    # Задача для тестирования модели
    test_task = PythonOperator(
        task_id='test_model',
        python_callable=test_model,
    )
    
    # Задача для вывода результатов
    print_results = BashOperator(
        task_id='print_results',
        bash_command='echo "Тестирование модели завершено"',
    )
    
    # Определение порядка выполнения задач
    test_task >> print_results 