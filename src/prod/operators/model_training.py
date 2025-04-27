import pandas as pd
import numpy as np
import boto3
import mlflow
from tqdm import tqdm
from datetime import datetime, timedelta
from src.utils import init_mlflow, log_experiment
import os 
import optuna
from sklearn.ensemble import GradientBoostingRegressor
from src.utils import init_mlflow, log_experiment, read_from_s3
from src.metric import FinEffect
from sklearn.metrics import mean_absolute_error
import pickle
from io import StringIO
import pandas as pd

import boto3

import logging


logger = logging.getLogger(__name__)


optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_boost_hyperparameters(
    features,
    target,
    n_trials: int = 100,
):

    metric = FinEffect()
    def objective(trial: optuna.Trial) -> float:

        n_estimators = trial.suggest_int('n_estimators', 100, 500)
        learning_rate = trial.suggest_categorical('learning_rate', [1e-1, 5e-2, 1e-2])
        max_depth = trial.suggest_int('max_depth', 3, 6)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        
        train = target.iloc[:-30:].index
        test = target.iloc[-30:].index

        len_train = len(train)
        X_train = features.iloc[:len_train,:]
        y_train = target.iloc[:len_train]

        preds = []
        for i in tqdm(range(5, len(test)+5, 5)):

            if i >= len(test):
                i = len(test)

            model =  GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            model.fit(X_train, y_train)

            X_train = features.iloc[:len_train+i,:]
            y_train = target.iloc[:len_train+i]
            prediction = model.predict(X_train)

            idx = 5 if i % 5 == 0 else i % 5
            preds.extend(prediction[-idx:])

        preds = np.array(preds).reshape(-1, )
        error = metric.model_effct(preds, target[test].values) * 100
        return error

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials
    )
    
    return study.best_params

def model_training_task(**context):
    """Task for model training and evaluation"""
    session = boto3.session.Session()
    s3_client = session.client(
            service_name='s3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
    )
    # Initialize MLflow
    init_mlflow(
        tracking_uri='http://host.docker.internal:5001',
        experiment_name='RegressBoost',
        s3_bucket=os.environ['MLFLOW_S3_BUCKET']
    )

    dag_run = context['dag_run']
    date = dag_run.conf.get('date') if dag_run and dag_run.conf else '2021-01-01'
    logger.info(f'Дата скоринга: {date}')
    date = datetime.strptime(date, '%Y-%m-%d')
    last_date = date
    test_border = date - timedelta(days=30)


    X = read_from_s3(
        s3_client,
        os.environ['AWS_BUCKET'],
        "prod/balance_data.csv",
        )
    X['Timestamp'] = pd.to_datetime(X['Timestamp'])
    X = X.set_index('Timestamp').bfill()


    y = read_from_s3(
        s3_client,
        os.environ['AWS_BUCKET'],
        "target/flow.csv",
        )
    y['Timestamp'] = pd.to_datetime(y['Timestamp'])
    y = y.set_index('Timestamp').bfill()['Balance']


    # Selection
    train = len(X[X.index < test_border])
    test = len(y[(y.index >= test_border)&((y.index <= last_date))])

    
    X_train, y_train = X.iloc[:train], y.iloc[:train]
    X_test, y_test = X.iloc[:-test], y.iloc[-test:]

    obj = s3_client.get_object(Bucket=os.environ['AWS_BUCKET'], Key='prod/top_features.pkl')
    top_features = pickle.loads(obj['Body'].read())


    # Fiting
    X = X[top_features]

    X_train, y_train = X.iloc[:train, :], y.iloc[:train]
    X_test, y_test = X.iloc[-test:], y.iloc[-test:]

    # подбор гиперпараметров
    params = tune_boost_hyperparameters(
        target = y_train,
        features = X_train,
        n_trials = 2
    )

    model =  GradientBoostingRegressor(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
    # обучение на базовом периоде
    model.fit(X_train, y_train)

    # онлайн дообучение (метрики)
    preds = model.predict(X_test)
    preds = np.array(preds).reshape(-1, )

    eff = FinEffect()

    metrics = {
        "mae": mean_absolute_error(y_test.to_numpy(), preds),
        "value": round(eff.model_effct(preds, y_test.to_numpy()) * 100, 10),
        "strike_share": np.sum(np.abs(y_test.to_numpy() - preds) >= 0.42) / len(y_test.to_numpy())
    }


    params = {
        "n_estimators" : params['n_estimators'], 
        "learning_rate": params['learning_rate'], 
        "max_depth" : params['max_depth'], 
        "min_samples_leaf" : params['min_samples_leaf'], 
    }
    artifacts = {
        'features': list(X_train.columns)
    }
    run_id = log_experiment(
        model=model,
        model_name="RegressBoost",
        params=params,
        metrics=metrics,
        artifacts=artifacts
    )
    logger.info(f'Новый run_id обученной модели: {run_id}')