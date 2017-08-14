from collections import OrderedDict

import numpy as np
import pandas as pd
from fbprophet import Prophet

from parallel_predictor import ParallelPredictor


class ProphetPredictor(ParallelPredictor):
    """
    In general behaves poorly by misreading trends. Score of ~56 compared to ~46 for median model.
    """
    def __init__(self, days_to_predict, disable_parallelism=False):
        super(ProphetPredictor, self).__init__(disable_parallelism=disable_parallelism)
        self.days_to_predict = days_to_predict

    def _serial_predict(self, X):
        log_x = np.log(X.fillna(0) + 1)
        future_index_df = pd.DataFrame({"ds": pd.date_range(X.index[-1] + pd.Timedelta(1, unit='d'), periods=self.days_to_predict)})
        predictions = OrderedDict()
        for c in X.columns:
            print u"predicting {}".format(c)
            prophet = Prophet(n_changepoints=5, uncertainty_samples=1000)
            singular_df = log_x[[c]]
            singular_df = singular_df.reset_index()
            singular_df = singular_df.rename(columns={c: "y", "index": "ds"})
            if singular_df.y.sum() > 0.0:
                prophet.fit(singular_df)
                future_df = prophet.predict(future_index_df)
                predictions[c] = (np.exp(future_df["yhat"]) - 1.).values
            else:
                predictions[c] = np.zeros((self.days_to_predict,))

        return pd.DataFrame(predictions, index=future_index_df.ds)

