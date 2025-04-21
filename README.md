# Balance Predictions ML Project

Проект для предсказания баланса с использованием MLflow для отслеживания экспериментов и хранения моделей в Yandex Cloud S3.

## Требования

- Python 3.9+
- Docker
- Docker Compose
- Доступ к Yandex Cloud S3

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd balance_predictions
```

2. Создайте виртуальное окружение и активируйте его:
```bash
bash setup_env.sh
source .venv/bin/activate  # для Linux/Mac
# или
.venv\Scripts\activate  # для Windows
```

3. Установите зависимости:
```bash
pip install -e .
```

4. Создайте файл .env с вашими учетными данными Yandex Cloud:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=ru-central1
MLFLOW_S3_BUCKET=your_bucket_name
MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
MLFLOW_ARTIFACT_ROOT=s3://your_bucket_name/mlflow
```

## Запуск MLflow

1. Запустите MLflow в Docker контейнере:
```bash
docker-compose up --build
```

2. Проверьте доступность MLflow UI:
   - Откройте http://localhost:5001 в браузере

## Использование

### Инициализация MLflow

```python
from src.utils import init_mlflow

init_mlflow(
    experiment_name="your_experiment_name",
    s3_bucket="your_bucket_name"
)
```

### Логирование эксперимента

```python
from src.utils import log_experiment

run_id = log_experiment(
    model=your_model,
    model_name="model_name",
    params={"param1": value1, "param2": value2},
    metrics={"metric1": value1, "metric2": value2},
    artifacts={"artifact1": data1, "artifact2": data2}
)
```

### Получение результатов эксперимента

```python
from src.utils import get_experiment_artifacts, get_experiment_metrics, get_experiment_params

# Получение артефактов
artifacts = get_experiment_artifacts(run_id)

# Получение метрик
metrics = get_experiment_metrics(run_id)

# Получение параметров
params = get_experiment_params(run_id)
```

## Пример использования

Смотрите `test_mlflow_example.py` для полного примера использования всех функций:

```bash
python test_mlflow_example.py
```

## Структура проекта

```
balance_predictions/
├── src/
│   ├── __init__.py
│   └── utils.py           # Утилиты для работы с MLflow
├── tests/
│   └── test_mlflow_example.py  # Пример использования
├── Dockerfile            # Конфигурация Docker для MLflow
├── docker-compose.yml    # Конфигурация Docker Compose
├── requirements.txt      # Зависимости Python
├── setup.py             # Конфигурация пакета
└── .env                 # Переменные окружения (не включать в репозиторий)
```

## Хранение данных

- Метаданные экспериментов хранятся в MLflow
- Модели и артефакты хранятся в Yandex Cloud S3
- Локальные копии артефактов хранятся в директории `mlruns/`

## Устранение неполадок

1. Если MLflow не запускается:
   - Проверьте, что порт 5001 свободен
   - Проверьте логи Docker контейнера: `docker-compose logs`

2. Если файлы не сохраняются в S3:
   - Проверьте учетные данные в .env
   - Убедитесь, что бакет существует и доступен
   - Проверьте права доступа к бакету

3. Если возникают проблемы с импортом:
   - Убедитесь, что пакет установлен: `pip install -e .`
   - Проверьте активацию виртуального окружения
