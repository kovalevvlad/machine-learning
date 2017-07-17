import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class LastValuePredictor(BaseEstimator, RegressorMixin):
    def __init__(self, days_to_predict):
        self.days_to_predict = days_to_predict

    def predict(self, X):
        last_day = X.index.values[-1]
        first_predicted_day = last_day + np.timedelta64(1, "D")
        predicted_days = pd.date_range(first_predicted_day, periods=self.days_to_predict)
        every_day_prediction = X.fillna(method="pad").fillna(0).iloc[-1]
        return pd.DataFrame({dt: every_day_prediction for dt in predicted_days}, index=X.columns).T

    def fit(self, X, y=None):
        return self
