import pandas as pd

from parallel_predictor import ParallelPredictor
import numpy as np


class WeekendWeekdayMedianPredictor(ParallelPredictor):
    def __init__(self, weeks_median, predicted_dates, disable_parallelism=False):
        """
        :param predicted_dates: either int or series of dates. If series of dates - predict for these. Otherwise use
                                the int to predict the number of days into the future.
        """
        super(WeekendWeekdayMedianPredictor, self).__init__(disable_parallelism=disable_parallelism)
        self.weeks_median = weeks_median
        self.predicted_dates = predicted_dates

    def _serial_predict(self, X):
        is_weekday = pd.Series(X.index.weekday.values, index=X.index) < 5

        median_weekday = X[is_weekday].tail(self.weeks_median * 5).median().values
        median_weekend = X[~is_weekday].tail(self.weeks_median * 2).median().values

        if type(self.predicted_dates) == int:
            predicted_dates = pd.date_range(X.index[-1] + pd.Timedelta(days=1), periods=self.predicted_dates)
        else:
            predicted_dates = self.predicted_dates

        future_is_weekday = pd.Series((predicted_dates.weekday < 5), index=predicted_dates)
        predicted_day_count = len(future_is_weekday)

        weekday_predictions = pd.DataFrame(np.tile(median_weekday, (predicted_day_count, 1)), index=predicted_dates, columns=X.columns)
        weekend_predictions = pd.DataFrame(np.tile(median_weekend, (predicted_day_count, 1)), index=predicted_dates, columns=X.columns)

        weekday_predictions[~future_is_weekday] = 0.0
        weekend_predictions[future_is_weekday] = 0.0
        weekend_weekday_merged = weekday_predictions + weekend_predictions
        return weekend_weekday_merged
