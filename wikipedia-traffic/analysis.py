# coding=utf-8
import datetime

import logging
import timeit

import pandas as pd
import numpy as np
from _pytest.mark import param
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

from data import train_df
from custom_time_series_split import CustomTimeseriesSplit
# from keras_predictor import KerasPredictor
from timeseries_predictor import TimeseriesPredictor
from smape import df_smape, np_smape
from time_series_cross_validation import cross_val_score, TSGridSearchCV
from matplotlib import pyplot as plt
from math_util import power_spectrum

from weekend_weekday_median_predictor import WeekendWeekdayMedianPredictor

predict_up_to = np.datetime64(datetime.datetime(2017, 3, 1))
last_date = train_df.index.values[-1]
days_to_predict = (pd.Timestamp(predict_up_to).to_pydatetime() - pd.Timestamp(last_date).to_pydatetime()).days
cv_strategy = CustomTimeseriesSplit(days_to_predict, 1, 490)

# params = [
#     {"de_trending_window": [7], "seasonal_threshold": [0.0075], "seasonality_estimation_window": [90]},
#     {"de_trending_window": [None], "seasonal_threshold": [None], "seasonality_estimation_window": [None]},
#     {"de_trending_window": [7], "seasonal_threshold": [0.005], "seasonality_estimation_window": [90]},
#     {"de_trending_window": [7], "seasonal_threshold": [None], "seasonality_estimation_window": [None]}
# ]
# logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
# result = TSGridSearchCV(
#     lambda args: WeekendWeekdayMedianPredictor(days_to_predict, 6, **args),
#     {
#         "periodicity_window": [30, 60, 90, 180],
#         "periodicity_threshold": [0.0, 0.02, 0.04, 0.06, 0.08, 0.1],
#     },
#     cv_strategy,
#     df_smape).fit(data)
# result.to_csv("grid_search2.csv")
#
# for i in range(3):
#     data = train_df.sample(n=1000, axis=1)
#     split_median = WeekendWeekdayMedianPredictor(6, 60, disable_parallelism=True)
#     lightgbm_predictor = LightgbmCompositePredictor(days_to_predict, 6, 1)
#     print "split median scored {}".format(cross_val_score(split_median, data, cv_strategy, df_smape).mean())
#     print "lightgbm scored {}".format(cross_val_score(lightgbm_predictor, data, cv_strategy, df_smape).mean())


# def run():

class SimpleModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X["median 28 for current day type"].values

# TODO: shift size
# TODO: try changing error functions
# TODO: play around with features - throw away useless and add aggregates and holidays
def run():
    results = []
    param_values = [10, 33, 100]
    for i in [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]:#[5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 12800, 25600]:
        # try for different series to reduce noise in the score plot
        for q in range(3):
            data = train_df.sample(n=i, axis=1)
            for param_value in param_values:

                # baseline = TimeseriesPredictor(days_to_predict, 1, 7, SimpleModel(), normalizing_period=7)
                # score = cross_val_score(baseline, data, cv_strategy, df_smape).mean()
                # print "baseline scored {:.2f}".format(score)

                # predictors = []
                # predictors.append(["baseline", TimeseriesPredictor(days_to_predict, 1, 7, SimpleModel(), normalizing_period=7)])
                # predictors.append(["lasso", TimeseriesPredictor(days_to_predict, 7, 4, Lasso(0.01), normalizing_period=7)])
                # predictors.append(["lgbm", TimeseriesPredictor(days_to_predict, 7, 4, LGBMRegressor(objective="huber"), one_hot=False, normalizing_period=7)])
                # predictors.append(["nn", TimeseriesPredictor(days_to_predict, 7, 4, KerasPredictor(), normalizing_period=7)])

                # for name, predictor in predictors:
                predictor = TimeseriesPredictor(days_to_predict, 1, 4, LGBMRegressor(objective="regression_l2", n_estimators=param_value), np_smape, one_hot=False, normalizing_period=21)
                # predictor = TimeseriesPredictor(days_to_predict, 7, 4, Ridge(alpha=param_value), np_smape, one_hot=True, normalizing_period=7)

                start = timeit.default_timer()
                train_scores, test_scores = cross_val_score(predictor, data, cv_strategy, df_smape)
                duration = timeit.default_timer() - start
                print "{} scored {:.2f}/{:.2f} with {} training values and param={}. Took {}.".format("gbt", train_scores.mean(), test_scores.mean(), i, param_value, duration)
                results.append(pd.DataFrame({"sample_size": [i] * len(train_scores), "train_score": train_scores, "test_score": test_scores, "param_value": param_value}))

    result_df = pd.concat(results)

    mean_test_score = result_df.groupby(["sample_size", "param_value"]).mean().test_score
    mean_train_score = result_df.groupby(["sample_size", "param_value"]).mean().train_score
    train_error = result_df.groupby(["sample_size", "param_value"]).std().train_score * 2
    test_error = result_df.groupby(["sample_size", "param_value"]).std().test_score * 2

    fig, axes = plt.subplots(nrows=3, ncols=3)
    all_axes = [ax for axes_row in axes for ax in axes_row]

    for param_value, ax in zip(param_values, all_axes):
        pd.DataFrame({"test score": mean_test_score.loc[:, param_value], "err": test_error.loc[:, param_value]}).plot(y="test score", yerr="err", ax=ax)
        pd.DataFrame({"train score": mean_train_score.loc[:, param_value], "err": train_error.loc[:, param_value]}).plot(y="train score", yerr="err", ax=ax)
        ax.set_title("param: {}".format(param_value))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set_xscale('log')

    plt.show()
    i = 0

#
# score_df = pd.DataFrame(results, columns=["name", "sample size", "score"])
# score_df["sample size"] = np.log(score_df["sample size"])
# mean = score_df.groupby(["name", "sample size"]).mean()
# mean.columns = ["mean score"]
# std = score_df.groupby(["name", "sample size"]).std()
# std.columns = ["std"]
# data = pd.concat([mean, std], axis=1, join="inner").reset_index().pivot()
# plt.show()
# i = 0

# logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
# import cProfile
# cProfile.runctx('run()', globals(), locals(), filename="profile.cpf")


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
run()

# result = {"period": [], "score": [], "error": []}
# periods = range(8, 60, 5)
# results = []
# for period in periods:
#     print "training with period {}".format(period)
#     scores = cross_val_score(MedianPredictor(days_to_predict, period), data, cv_strategy, smape)
#     results.append((scores.mean(), scores.std() * 2))
#
# scores, errors = zip(*results)
# pd.DataFrame(zip(periods, scores, errors), columns=["period", "score", "error"]).plot(kind="line", x="period", y="score", yerr="error")
# plt.show()

#
# arima_model = SeasonalArimaPredictor(days_to_predict, 0, 0, 1, 1, 1, 1)
# arima_model2 = SeasonalArimaPredictor(days_to_predict, 0, 0, 0, 1, 1, 1)
#
# arima_scores = cross_val_score(arima_model, data, cv_strategy, smape)
#
# print "{}: {:.2f} +- {:.2f}".format("arima", arima_scores.mean(), arima_scores.std())

# for column in data.columns:
#     print column
#     series = np.log(data[[column]] + 1)
#     train = series.iloc[:-60]
#     test = series.iloc[-60:]
#     predictions = WeekendWeekdayMedianPredictor(days_to_predict, 6).predict(train)
#     diff = (test - predictions)
#     spectrum = power_spectrum(diff, 49)
#
#     fig, axs = plt.subplots(1, 2)
#     diff.plot(ax=axs[0])
#     spectrum.plot(ax=axs[1])
#
#     plt.show()
#     i=0

# arima_model.predict(data)
# cv_scores_arima = cross_val_score(arima_model, data, cv_strategy, smape)
# print "{}: {:.2f} +- {:.2f}".format("arima", cv_scores_arima.mean(), cv_scores_arima.std())

# def run():
#     p = SeasonalArimaPredictor(days_to_predict, 12, 0, 0, 1, 1, 1, 1, seasonal_threshold=0.0075, seasonality_estimation_window=180, de_trending_window=7, disable_parallelism=True).predict(data)
#
# logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)
# import cProfile
# cProfile.runctx('run()', globals(), locals(), filename="profile.cpf")


# median_model = MedianPredictor(days_to_predict, 11)
# arima_model = SeasonalArimaPredictor(days_to_predict, 0, 0, 1, 1, 1, 1)
# median_scores = cross_val_score(median_model, data, cv_strategy, smape)
# arima_scores = cross_val_score(arima_model, data, cv_strategy, smape)
# print "{}: {:.2f} +- {:.2f}".format("median", median_scores.mean(), median_scores.std())
# print "{}: {:.2f} +- {:.2f}".format("arima", arima_scores.mean(), arima_scores.std())
