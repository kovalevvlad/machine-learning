# coding=utf-8
import datetime

import timeit

import pandas as pd
import numpy as np
from data import train_df
from custom_time_series_split import CustomTimeseriesSplit
from keras_predictor import KerasPredictor
from timeseries_predictor import TimeseriesPredictor
from smape import df_smape, np_smape
from matplotlib import pyplot as plt
predict_up_to = np.datetime64(datetime.datetime(2017, 3, 1))
last_date = train_df.index.values[-1]
days_to_predict = (pd.Timestamp(predict_up_to).to_pydatetime() - pd.Timestamp(last_date).to_pydatetime()).days
cv_strategy = CustomTimeseriesSplit(days_to_predict, 1, 490)


class SimpleModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X["median 28 for current day type"].values


def run():
    for pages_to_process in [15000]:
        data = train_df.sample(n=pages_to_process, axis=1, random_state=42)
        history_dfs = dict()
        param_values = [1]
        (train_data, test_data), = tuple(cv_strategy.split(data))
        median_model_score = df_smape(TimeseriesPredictor(days_to_predict, 1, 1, SimpleModel(), np_smape).predict(train_data), test_data)
        for param_value in param_values:
            predictor = TimeseriesPredictor(
                days_to_predict,
                3,
                60,
                KerasPredictor((60, 50, 35, 10, 8, 5, 5, 5, 5), learning_rate=0.003, decay=7e-7),
                np_smape,
                one_hot=True)
            start = timeit.default_timer()
            prediction = predictor.predict(train_data, validation_split=0.1, epochs=1000, batch_size=256)
            smape_score = df_smape(test_data, prediction)
            duration = timeit.default_timer() - start
            print "took {} with {}".format(duration, param_value)
            print "SMAPE = {} vs Median Model = {}".format(smape_score, median_model_score)
            history_df = pd.DataFrame({
                "train": predictor.fitting_result.history["loss"],
                "test": predictor.fitting_result.history["val_loss"]
            })
            history_dfs[param_value] = history_df

        fig, axes = plt.subplots(nrows=3, ncols=3)
        all_axes = [ax for axes_row in axes for ax in axes_row]

        for param_value, ax in zip(param_values, all_axes):
            df = history_dfs[param_value]
            df.plot(ax=ax)
            ax.set_title("param: {}".format(param_value))
            ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            ax.set_ylim(43, 68)

        plt.show()

run()
