Групповой проект в рамках предмета "Анализ данных с временной структурой"


## Структура проекта
balance_predictions/
├── airflow_dags/                # DAG’и для ETL и ретренинга (пусть пока так)
├── src/
│   ├── config.py                # параметры и константы
│   ├── data_ingestion.py        # модуль сбора из БД и API (пусть пока так, для масштабирования нужно будет)
│   ├── feature_engineering.py   # генерация таргет‑фичей и внешних
│   ├── feature_selection.py     # filter, embedded, wrapper методы
│   ├── model_trainer.py         # тренировка, CV, гипертюнинг
│   ├── drift_detector.py        # ADWIN / River‑модуль
│   ├── retraining.py            # логика автодообучения
│   ├── inference_service.py     # FastAPI‑эндпойнты для прогноза
│   └── utils.py                 # вспомогательные функции
├── mlflow/                      # конфиг MLflow (server, ui)
├── Dockerfile                   # сборка контейнера
├── requirements.txt
├── helm_chart/                  # шаблон для Kubernetes (пусть будет на всякий)
└── README.md
