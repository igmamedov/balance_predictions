import phik
import pandas as pd

class FeatureSelection:
    """
    Отбор признаков

    Args:
        X (pd.DataFrame): предлашаемые признаки
        y (pd.DataFrame): таргет
    """
    def __init__(self, X, y):
        assert len(X) == len(y), "X and y must be the same length"

        self.X = X
        self.y = y
        self.best_features = None

    def select_best(self, n=20, threshold=0.1):
        """
        Args:
            n (int): макс кол-во фичей

        Returns:
            top_features (list[str]): список лучших признаков
        """
        df = pd.concat([self.y, self.X], axis=1).phik_matrix()
        features = df[df[self.y.name] >= threshold][self.y.name].sort_values(ascending=False)
        top_features = list(features[1:n+1].index)
        self.best_features = top_features
        return top_features