import pandas as pd
from parallel_predictor import ParallelPredictor
import numpy as np


class MedianPredictor(ParallelPredictor):
    def __init__(self, days_to_predict, period, disable_parallelism=False):
        super(MedianPredictor, self).__init__(disable_parallelism=disable_parallelism)
        self.days_to_predict = days_to_predict
        self.period = period

    def _serial_predict(self, X):
        median = X.tail(self.period).median().values

        if type(self.days_to_predict) == int:
            predicted_dates = pd.date_range(X.index[-1] + pd.Timedelta(1, unit='d'), periods=self.days_to_predict)
        else:
            predicted_dates = self.days_to_predict

        predictions = np.tile(median, (len(predicted_dates), 1))
        float_predictions = pd.DataFrame(predictions, columns=X.columns, index=predicted_dates).fillna(0)
        return float_predictions.round()
