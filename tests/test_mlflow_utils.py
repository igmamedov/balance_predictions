import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from utils import init_mlflow, log_experiment, get_experiment_artifacts

# Инициализация MLflow
init_mlflow(
    experiment_name="test_experiment_utils",
    s3_bucket="balance-predictions"
)

# Создание тестовых данных
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Создание и обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказания для оценки
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# Параметры модели
params = {
    "model_type": "LinearRegression",
    "fit_intercept": True
}

# Метрики модели
metrics = {
    "mse": mse
}

# Дополнительные артефакты
artifacts = {
    "X": X,
    "y": y,
    "y_pred": y_pred
}

# Логирование эксперимента
run_id = log_experiment(
    model=model,
    model_name="test_linear_model",
    params=params,
    metrics=metrics,
    artifacts=artifacts
)

print(f"Эксперимент успешно завершен! Run ID: {run_id}")

# Получение артефактов эксперимента
artifacts = get_experiment_artifacts(run_id)
print("\nПолученные артефакты:")
for name, artifact in artifacts.items():
    print(f"- {name}") 