# Balance Predictions ML Project

Проект для предсказания баланса с использованием MLflow для отслеживания экспериментов и хранения моделей в Yandex Cloud S3.

## Требования

- Python 3.9+
- Docker
- Docker Compose
- Доступ к Yandex Cloud S3

## Установка

1. Создайте файл `.env` в корне проекта со следующими переменными:
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=ru-central1
MLFLOW_S3_BUCKET=balance-predictions
```

2. Запустите MLflow сервер:
```bash
docker compose up --build
```

3. MLflow UI будет доступен по адресу: http://localhost:5001

## Запуск Airflow

1. Создайте необходимые директории:
```bash
mkdir -p dags logs plugins config
```

2. Инициализируйте Airflow:
```bash
docker compose -f docker-compose-airflow.yml up airflow-init
```

3. Запустите Airflow:
```bash
docker compose -f docker-compose-airflow.yml up --build
```

4. Airflow UI будет доступен по адресу: http://localhost:8080
   - Логин: airflow
   - Пароль: airflow

## Структура проекта

```
balance_predictions/
├── src/
│   ├── __init__.py
│   └── utils.py           # Утилиты для работы с MLflow
├── tests/
│   └── test_mlflow_utils.py    # Тесты утилит MLflow
├── dags/
│   └── test_mlflow_dag.py      # Тестовый DAG для Airflow
├── logs/                       # Логи Airflow 
├── plugins/                    # Плагины Airflow
├── config/                     # Конфигурационные файлы
├── Dockerfile                  # Конфигурация Docker для MLflow
├── docker-compose.yml          # Конфигурация Docker Compose для MLflow
├── docker-compose-airflow.yml  # Конфигурация Docker Compose для Airflow
├── requirements.txt            # Зависимости Python
├── setup.py                    # Конфигурация пакета
└── README.md                   # Документация проекта
```

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

## Запуск Airflow

1. Создайте необходимые директории:
```bash
mkdir -p dags logs plugins config
```

2. Инициализируйте Airflow (только при первом запуске):
```bash
docker compose -f docker-compose-airflow.yml up airflow-init
```

3. Запустите Airflow в Docker контейнере:
```bash
docker compose -f docker-compose-airflow.yml up --build
```

4. Проверьте доступность Airflow UI:
   - Откройте http://localhost:8080 в браузере
   - Логин: airflow
   - Пароль: airflow

### Структура Airflow

- `dags/` - директория для DAG файлов
  - `test_mlflow_dag.py` - пример DAG'а для тестирования модели
- `logs/` - логи выполнения задач
- `plugins/` - пользовательские плагины
- `config/` - конфигурационные файлы

### Пример использования Airflow

1. Убедитесь, что MLflow запущен и содержит обученную модель:
```bash
docker compose up --build
python tests/test_mlflow_utils.py
```

2. Запустите Airflow:
```bash
docker compose -f docker-compose-airflow.yml up --build
```

3. Откройте Airflow UI (http://localhost:8080) и найдите DAG "test_mlflow_model"

4. Включите DAG и запустите его:
   - Нажмите на переключатель слева от имени DAG'а для активации
   - Нажмите на кнопку "Trigger DAG" для запуска

5. Проверьте результаты:
   - В Airflow UI: Graph View и Task Instance Details
   - В MLflow UI (http://localhost:5001): новый эксперимент с результатами теста

### Описание тестового DAG'а

`test_mlflow_dag.py` демонстрирует:
- Загрузку существующей модели из MLflow
- Создание тестовых данных
- Выполнение предсказаний
- Логирование результатов обратно в MLflow

Результаты включают:
- Предсказания на новых данных
- Метрики оригинальной модели
- Параметры теста
- Тестовые данные и предсказания как артефакты

## Использование

### Инициализация MLflow

```python
from src.utils import init_mlflow

init_mlflow(
    tracking_uri='http://localhost:5001',
    experiment_name="your_experiment",
    s3_bucket="balance-predictions"
)
```

### Логирование эксперимента

```python
from src.utils import log_experiment

run_id = log_experiment(
    model=your_model,
    model_name="model_name",
    params={"param1": "value1"},
    metrics={"metric1": 0.95},
    artifacts={"artifact1": data}
)
```

### Получение артефактов
```python
from src.utils import get_artifacts_by_run_name

# Получение артефактов по имени запуска
artifacts = get_artifacts_by_run_name("run_name", "experiment_name")
model = artifacts["model"]

# Получение артефактов по ID запуска
artifacts = get_experiment_artifacts("run_id")
model = artifacts["model"]
```

## Пример использования

Смотрите `test_mlflow_example.py` для полного примера использования всех функций:

```bash
python test_mlflow_example.py
```

### Описание компонентов

- `src/`: Исходный код проекта
  - `utils.py`: Утилиты для работы с MLflow (инициализация, логирование, получение результатов)
  
- `tests/`: Тесты и примеры использования
  - `test_mlflow_utils.py`: Тесты для проверки функциональности утилит
  - `test_mlflow_example.py`: Пример использования MLflow с реальными данными

- `dags/`: DAG'и Airflow
  - `test_mlflow_dag.py`: Пример DAG'а для тестирования моделей из MLflow

- `logs/`: Логи выполнения задач Airflow

- `plugins/`: Пользовательские плагины для расширения функциональности Airflow

- `config/`: Конфигурационные файлы для настройки компонентов

- Docker файлы:
  - `Dockerfile`: Конфигурация контейнера MLflow
  - `docker-compose.yml`: Настройка сервисов MLflow
  - `docker-compose-airflow.yml`: Настройка сервисов Airflow

- Конфигурационные файлы:
  - `requirements.txt`: Зависимости Python
  - `setup.py`: Конфигурация пакета
  - `.env`: Переменные окружения (не включены в репозиторий)

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

## Работа с Airflow

### Структура DAG
```python
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'your_dag',
    default_args=default_args,
    description='Описание DAG',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    
    task = PythonOperator(
        task_id='task_name',
        python_callable=your_function,
    )
```

### Мониторинг DAG
1. Откройте Airflow UI (http://localhost:8080)
2. Найдите ваш DAG в списке
3. Используйте кнопки для управления DAG:
   - Trigger DAG - запустить DAG вручную
   - Pause/Unpause - приостановить/возобновить DAG
   - Graph View - просмотр графа задач
   - Tree View - просмотр истории запусков

### Логирование
- Логи задач доступны в Airflow UI
- Логи MLflow доступны в MLflow UI
- Файлы логов находятся в директории `logs/`

## Взаимодействие MLflow и Airflow

1. Airflow использует MLflow для:
   - Загрузки моделей
   - Логирования экспериментов
   - Хранения артефактов

2. MLflow хранит:
   - Модели
   - Метрики
   - Параметры
   - Артефакты

3. Пример использования в DAG:
```python
def test_model():
    init_mlflow(
        tracking_uri='http://host.docker.internal:5001',
        experiment_name="test_experiment",
        s3_bucket="balance-predictions"
    )
    
    artifacts = get_artifacts_by_run_name("model_run", "test_experiment")
    model = artifacts["model"]
    
    # Использование модели
    predictions = model.predict(data)
    
    # Логирование результатов
    log_experiment(
        model=model,
        model_name="test_model",
        metrics={"accuracy": 0.95},
        artifacts={"predictions": predictions}
    )
```

## Устранение неполадок

1. Если MLflow недоступен:
   - Проверьте, запущен ли контейнер MLflow
   - Проверьте порт 5001
   - Проверьте переменные окружения

2. Если Airflow недоступен:
   - Проверьте, запущены ли все контейнеры Airflow
   - Проверьте порт 8080
   - Проверьте логи контейнеров

3. Если DAG не запускается:
   - Проверьте синтаксис DAG
   - Проверьте зависимости
   - Проверьте логи задач

4. Если модель не загружается:
   - Проверьте имя эксперимента
   - Проверьте имя запуска
   - Проверьте наличие модели в MLflow
