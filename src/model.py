import numpy as np
import pmdarima as pm

import warnings
warnings.filterwarnings("ignore")


class Estimator:
    """
    Модуль для предсказания баланса

    Args:
        m (int): сезонность
        start_q (int): минимальный парамаметр q для подбора
        max_q (int): максимальный парамаметр q для подбора
        start_p (int): минимальный парамаметр p для подбора
        max_p (int): максимальный парамаметр p для подбора
    """
    def __init__(self, m , start_q, max_q, start_p, max_p):
        self.m = m
        self.start_q = start_q
        self.max_q = max_q
        self.start_p = start_p
        self.max_p = max_p
        self.last_prediction = None
        self.model = None


    def fit(self, X_train, y_train):
        """
        Обучение модели и подбор параметров

        Args:
            X_train (pd.DataFrame): экзагенные переменные
            y_train (pd.DataFrame): тагрег
        """
        self.model=pm.auto_arima(y = y_train,
                    X = X_train,
                    start_p=self.start_p,
                    max_p=self.max_p,
                    start_q=self.start_q,
                    max_q=self.max_q,
                    test='adf',
                    m=self.m,
                    seasonal=True,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True)

    def predict(self, X_test):
        """
        Делаем предсказание на новых данных

        Args:
            X_test (pd.DataFrame): новые экзогенные атрибуты
        """
        assert self.model is not None, "model is not fitted"
        forecast = self.model.predict(X=X_test, n_periods=1)
        self.last_prediction = forecast
        return forecast

    def update(self, X_test, last_value=None):
        """
        Делаем предсказание на новых данных

        Args:
            last_value (float): обновленное знаечние за прошлый предсказанный период (если нет, то берем прощлый прогноз)
            X_test (pd.DataFrame): обновленное значение X (1, n_features)
        """
        assert self.model is not None, "model is not fitted"

        if last_value is None:
            self.model.update(np.array([self.last_prediction]), X_test)
        else:
            self.model.update(np.array([last_value]), X_test)