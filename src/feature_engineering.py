from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures
from typing import Optional
import holidays
import numpy as np
import pandas as pd


class FeatureEngineering:
    def __init__(self, df:pd.DataFrame, calendar_df:Optional[pd.DataFrame] = None, n_lags:int=10, attrs:list[str] = [], tax_days:list[int] = [], alphas: list[float] = []):
        """
        Класс для генерации факторов времянного ряда

        Args:
            df (pd.DataFrame): твблицв с данными рядов (обязательно должен быть индeкс datetime, freq=D)
            calendar_df (pd.DataFrame): данные с рабочими днями компаниями (обязательно должен быть индeкс datetime, freq=D)
            n_lags (int): кол-во лагов для генерации
            attrs (list[str]): списки полей по которым гереируются признаки
            tax_days (list[int]): список дней налогов (выбирается первый рабочий день после)
            alphas (list[float]): список альф для экспоненциального сгладивания
        """

        assert hasattr(df.index, 'freq'), "df index should be datetime"
        if df.index.freq != "D":
            df = df.asfreq('D')
            df = df.fillna(0)


        assert hasattr(calendar_df.index, 'freq'), "calendar_df index should be datetime"
        if calendar_df.index.freq != "D":
            calendar_df = calendar_df.asfreq('D')
            calendar_df = calendar_df.fillna(0)

        self.df = df
        self.calendar_df = calendar_df
        self.attrs = attrs
        self.n_lags = n_lags
        self.tax_days = tax_days
        self.alphas = alphas

    def generate_lag_fetures(self, attr):
        """
        Генерация лаговых переменных

        Args:
            attr (str): выбранный аттрибут

        Returns:
            lag_features (pd.DataFrame): сгенерированные переменные
        """
        lag_processor = LagFeatures(periods=[x for x in range(1, self.n_lags)], drop_original=True, fill_value=0)
        lag_features = lag_processor.fit_transform(self.df[attr].to_frame())
        return lag_features

    def generate_slide_window_fetures(self, attr):
        """
        Генерация переменных скользящих окон

        Args:
            attr (str): выбранный аттрибут

        Returns:
            sliding_features (pd.DataFrame): сгенерированные переменные
        """
        sliding_window_processor = WindowFeatures(
            window=["2D", "5D", "10D", "20D", "60D"], 
            functions=["mean", "median", "min", "max", "std", "skew", "kurt"], 
            freq="1D",
        )

        sliding_features = sliding_window_processor.fit_transform(self.df[attr].to_frame())
        sliding_features = sliding_features.fillna(0).drop(columns=[attr], axis=1)
        return sliding_features
    
    def generate_expanding_window_features(self, attr):
        """
        Генерация переменных расширяюших окон

        Args:
            attr (str): выбранный аттрибут

        Returns:
            expanding_features (pd.DataFrame): сгенерированные переменные
        """
        expanding_window_processor = ExpandingWindowFeatures( 
            functions=["mean", "median", "min", "max", "std", "skew", "kurt"], 
            freq="1D",
        )

        expanding_features = expanding_window_processor.fit_transform(self.df[attr].to_frame())
        expanding_features = expanding_features.fillna(0).drop(columns=[attr], axis=1)
        return expanding_features


    def generate_ewma_features(self, attr, alpha):
        """
        Генерация переменных экспоненциального сгладивания

        Args:
            attr (str): выбранный аттрибут
            alpha (float): сила сглаживания 

        Returns:
            ewma_features (pd.DataFrame): сгенерированные переменные
        """
        ewma_data = self.df[attr].ewm(alpha=alpha)

        ewma_features = pd.DataFrame({
            f"{attr}_ewma_{alpha}_mean":ewma_data.mean(),
            f"{attr}_ewma_{alpha}_std":ewma_data.std(),
        })
        ewma_features = ewma_features.fillna(0)
        return ewma_features
    
    def generate_calendar_features(self):
        """
        Генерация календарных переменных

        Returns:
            calendar_features (pd.DataFrame): сгенерированные переменные
        """

        cal_h = holidays.RU()
        calendar_features = pd.DataFrame({
            'year':self.df.index.year,
            'month':self.df.index.month,
            'day':self.df.index.day,
            'day_of_week':self.df.index.day_of_week,
            'is_holiday':pd.Series(self.df.index).apply(lambda x: x in cal_h).astype(int)
        })
        calendar_features.index = self.df.index

        if self.calendar_df is not None:
            calendar_features['is_not_working_day'] = pd.merge(self.df, self.calendar_df, left_index=True, right_index=True, how='left')['not_working_day']

        else:
            calendar_features['is_not_working_day'] = np.min(calendar_features['is_holiday'] + calendar_features['day_of_week'].isin([5,6]).astype(int), 1)

        calendar_features_year = pd.get_dummies(calendar_features['year'], prefix='is_year').astype(int)
        calendar_features_month = pd.get_dummies(calendar_features['month'], prefix='is_month').astype(int)
        calendar_features_day = pd.get_dummies(calendar_features['day'], prefix='is_day').astype(int)
        calendar_features_weekday = pd.get_dummies(calendar_features['day_of_week'], prefix='is_weekday').astype(int)

        calendar_features = pd.concat([calendar_features, calendar_features_year, calendar_features_month, calendar_features_day, calendar_features_weekday], axis=1)
        calendar_features = calendar_features.drop(columns=['year', 'month', 'day', 'day_of_week'])
        return calendar_features
    
    def generate_tax_features(self, day):
        """
        Генерация налоговый переменных

        Args:
            day (int): число налога (выбирается ближайший следующий рабочий день)

        Returns:
            tax_features (pd.DataFrame): сгенерированные переменные
        """
        tax_features = pd.DataFrame(index=self.df.index)
        tax_features['week_day'] = tax_features.index.day_of_week

        tax_name = f'is_tax_{day}'
        tax_features[tax_name] = 0
        
        for year in tax_features.index.year.unique():
            for month in tax_features.index.month.unique():
                next_days = tax_features[
                    (tax_features.index.day >= day)&\
                    (tax_features.index.year == year)&\
                    (tax_features.index.month == month)   
                ]
        
                next_working_day = next_days[next_days['week_day'].isin([0,1,2,3,4])].first_valid_index()
                if next_working_day is not None:
                    tax_features.loc[next_working_day, tax_name] = 1
    
        tax_features = tax_features.drop(columns=['week_day'], axis=1)
        return tax_features

    def generate_features(self):
        """
        Генерация переменных для выбранных рядов

        Returns:
            featured_df (pd.DataFrame): сгенерированные переменные
        """
        featured_df = pd.DataFrame(index=self.df.index)

        calendar_features = self.generate_calendar_features()
        featured_df = pd.merge(featured_df, calendar_features, left_index=True, right_index=True, how='left')

        for day in self.tax_days:
            tax_features = self.generate_tax_features(day)
            featured_df = pd.merge(featured_df, tax_features, left_index=True, right_index=True, how='left')

        for attr in self.attrs:
            lag_fetures = self.generate_lag_fetures(attr)
            featured_df = pd.merge(featured_df, lag_fetures, left_index=True, right_index=True, how='left')

            slide_window_fetures = self.generate_slide_window_fetures(attr)
            featured_df = pd.merge(featured_df, slide_window_fetures, left_index=True, right_index=True, how='left')

            expanding_window_features = self.generate_expanding_window_features(attr)
            featured_df = pd.merge(featured_df, expanding_window_features, left_index=True, right_index=True, how='left')

            for alpha in self.alphas:
                ewma_features = self.generate_ewma_features(attr, alpha)
                featured_df = pd.merge(featured_df, ewma_features, left_index=True, right_index=True, how='left')

        return featured_df