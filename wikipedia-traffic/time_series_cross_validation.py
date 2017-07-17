import pandas as pd
import numpy as np


class WalkForwardCV:
    def __init__(self, cv):
        self.cv = cv


def np_datetime_to_date_str(d):
    return pd.Timestamp(d).to_pydatetime().strftime("%Y-%m-%d")


def cross_val_score(model, X, cv, score):
    scores = []
    for train, test in cv.split(X):
        print "running CV score with train {}..{} and test {}..{}".format(
            np_datetime_to_date_str(train.index.values[0]),
            np_datetime_to_date_str(train.index.values[-1]),
            np_datetime_to_date_str(test.index.values[0]),
            np_datetime_to_date_str(test.index.values[-1]))
        model.fit(train)
        predictions = model.predict(train)
        result = score(predictions, test)
        print "CV round yielded {}".format(result)
        scores.append(result)
    return pd.Series(scores)
