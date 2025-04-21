import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Установка URI для MLflow
mlflow.set_tracking_uri("http://localhost:5001")

# Настройка S3
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://storage.yandexcloud.net'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'

# Создание эксперимента
experiment_name = "test_experiment_2"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Создание тестовых данных
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Создание и обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказания для оценки
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# Начало эксперимента
with mlflow.start_run(run_name="test_linear_regression_tt") as run:
    # Логирование параметров
    mlflow.log_param("model_type", "LinearRegression")
    
    # Логирование метрик
    mlflow.log_metric("mse", mse)
    
    # Логирование модели
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="test_linear_model"
    )
    
    # Логирование дополнительной информации
    mlflow.log_dict({"description": "Тестовая модель линейной регрессии"}, "model_info.json")
    
    print(f"Эксперимент успешно завершен! Run ID: {run.info.run_id}")
    print(f"MSE: {mse}")
    print(f"Коэффициенты модели: {model.coef_}")
    print(f"Перехват: {model.intercept_}") 