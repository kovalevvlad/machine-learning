import logging
import timeit

import pandas as pd
import numpy as np
import itertools


class WalkForwardCV:
    def __init__(self, cv):
        self.cv = cv


def np_datetime_to_date_str(d):
    return pd.Timestamp(d).to_pydatetime().strftime("%Y-%m-%d")


def cross_val_score(model, X, cv, score, **kwargs):
    scores = []
    for train, test in cv.split(X):
        logging.debug("running CV score with train {}..{} and test {}..{}".format(
            np_datetime_to_date_str(train.index.values[0]),
            np_datetime_to_date_str(train.index.values[-1]),
            np_datetime_to_date_str(test.index.values[0]),
            np_datetime_to_date_str(test.index.values[-1])))
        predictions = model.predict(train, **kwargs)
        result = score(test, predictions)
        logging.debug("CV round yielded {}".format(result))
        scores.append(result)
    return pd.Series(scores)


class TSGridSearchCV:
    def __init__(self, model_constructor, params, cv, score):
        if type(params) == dict:
            self.params = [params]
        else:
            self.params = params

        self.model_constructor = model_constructor
        self.cv = cv
        self.score = score

    def fit(self, X):
        param_names = list(set(k for param_dict in self.params for k in param_dict.keys()))
        results = {column: [] for column in param_names + ["score", "std", "duration"]}
        for param_dict in self.params:
            param_values = [param_dict[key] for key in param_names]
            param_combinations = list(itertools.product(*param_values))
            for param_combination in param_combinations:
                param_dict = dict(zip(param_names, param_combination))
                logging.debug("running with {}".format(param_dict))
                start = timeit.default_timer()
                cv_scores = cross_val_score(self.model_constructor(param_dict), X, self.cv, self.score)
                duration = timeit.default_timer() - start
                for param_name, param_value in param_dict.items():
                    results[param_name].append(param_value)
                results["score"].append(cv_scores.mean())
                results["std"].append(cv_scores.std())
                results["duration"].append(duration)
        return pd.DataFrame(results)
