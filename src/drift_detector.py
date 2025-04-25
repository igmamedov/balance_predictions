import pandas as pd
import numpy as np
import ruptures as rpt

class ChangePoint:
    """
    Обнаружение разладки

    dataset (pd.series): текущий ряд
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.model = rpt.Pelt(model="l1", min_size=10, jump=10)
        self.result = [0]

    def _detect_change_point(self, val, date):
        """
        Онлайн детектирование

        Args:
            val (float): новое значение ряда
            date (): дата нового значения

        Return:
            flag (bool): флаг была ли замечена новая раздадка
        """
        new_row = pd.Series({date: val})
        self.dataset = pd.concat([self.dataset, new_row])

        current_set = np.array(self.dataset.iloc[self.result[-1]:])

        point = list(np.array(self.result[-1]) + np.array(self.model.fit(current_set).predict(pen=4)[:-1]))


        if (len(point) > 0) and (max(point) > max(self.result)):
            flag = True
        else:
            flag = False

        self.result.extend(point)
        self.result = list(np.unique(sorted(self.result)))
        return flag

    def calc_statistics(self,  val, date, window=5):
        """
        Пересчет статистик внутри каждой разладки

        Args:
            val (float): новое значение ряда
            date (pd.Timestamp): дата нового значения
        Return:
            flag (bool): флаг была ли замечена новая раздадка
            stats (pd.DataFrame): таблица с признаками для каждой группы
        """
        flag = self._detect_change_point(val, date)
            
        window_data = []
        for i, start in enumerate(self.result):
            end = self.result[i + 1] if i + 1 < len(self.result) else len(self.dataset)
            segment = self.dataset.iloc[start:end]
            rolling_mean = segment.rolling(window=window, min_periods=1).mean()
            rolling_var = segment.rolling(window=window, min_periods=1).std()

            df = pd.DataFrame({
                'index': segment.index,
                'value': segment.values,
                'rolling_mean': rolling_mean.values,
                'rolling_std': rolling_var.values,
                'segment_id': i
            })
            window_data.append(df)

        stats = pd.concat(window_data, ignore_index=True).set_index('index').fillna(0)
        return flag, stats