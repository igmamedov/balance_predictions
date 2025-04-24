# Balance Predictions ML Project

Проект для предсказания баланса с использованием MLflow для отслеживания экспериментов и хранения моделей в Yandex Cloud S3.

## Требования

- Python 3.9+
- Docker
- Docker Compose
- Доступ к Yandex Cloud S3

## Установка


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
