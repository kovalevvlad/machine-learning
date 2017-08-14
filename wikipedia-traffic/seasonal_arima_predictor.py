# coding=utf-8
from collections import OrderedDict
import logging
import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math_util import weekly_power_spectrum
from median_predictor import MedianPredictor
from parallel_predictor import ParallelPredictor


class SeasonalArimaPredictor(ParallelPredictor):
    def __init__(self,
                 days_to_predict,
                 median_period,
                 p,
                 d,
                 q,
                 sp,
                 sd,
                 sq,
                 disable_parallelism=False,
                 seasonal_threshold=None,
                 seasonality_estimation_window=None,
                 de_trending_window=None):
        super(SeasonalArimaPredictor, self).__init__(disable_parallelism=disable_parallelism)
        self.days_to_predict = days_to_predict
        self.median_period = median_period
        self.p = p
        self.d = d
        self.q = q
        self.sp = sp
        self.sd = sd
        self.sq = sq
        self.seasonal_threshold = seasonal_threshold
        self.seasonality_estimation_window = seasonality_estimation_window
        self.de_trending_window = de_trending_window

    def _serial_predict(self, X):
        assert X.isnull().sum().sum() == 0
        assert (X < 0).sum().sum() == 0
        assert ((X == np.inf) | (X == -np.inf)).sum().sum() == 0

        log_x = np.log(X.fillna(0) + 1)
        log_median_predictions = np.log(MedianPredictor(self.days_to_predict, self.median_period, disable_parallelism=True).predict(X) + 1)
        forecast_index = pd.date_range(X.index[-1] + pd.Timedelta(1, unit='d'), periods=self.days_to_predict)

        predictions = OrderedDict()
        for column in log_x.columns:
            ts = log_x[column]
            if len(ts.dropna().unique()) == 1:
                # ARIMA breaks if all values are the same
                predictions[column] = np.full(forecast_index.shape, ts.values[0])
            else:
                if self.seasonal_threshold is None or weekly_power_spectrum(ts, self.seasonality_estimation_window) > self.seasonal_threshold:
                    try:
                        forecast = self._arima_prediction(ts)
                        predictions[column] = forecast
                        # Arima is bad at predicting level, use median predictor instead
                        predictions[column] = predictions[column] - predictions[column].mean() + log_median_predictions[column].iloc[-1]
                    except LinAlgError:
                        # This appears to happen very rarely for an unknown reason...
                        predictions[column] = log_median_predictions[column]
                        logging.error(u"LinalgError for {}".format(column))
                else:
                    predictions[column] = log_median_predictions[column]

        predictions = pd.DataFrame(predictions, index=forecast_index)
        predictions = np.exp(predictions) - 1
        # If mean == 0 we get nans. Fix it.
        return predictions.fillna(0)

    def _arima_prediction(self, ts):
        params = dict(p=self.p, d=self.d, q=self.q, sp=self.sp, sd=self.sd, sq=self.sq)
        de_trended = ts - ts.rolling(self.de_trending_window).median() if self.de_trending_window is not None else ts
        # enforce_* = False gives a massive speed up and doesn't hurt predictive performance (impirically tested)
        arima = SARIMAX(
            de_trended,
            order=(params["p"], params["d"], params["q"]),
            seasonal_order=(params["sp"], params["sd"], params["sq"], 7),
            enforce_stationarity=False,
            enforce_invertibility=False)
        fitting_result = arima.fit(disp=0)
        start = len(de_trended)
        return fitting_result.predict(start=start, end=start + self.days_to_predict, dynamic=True)
