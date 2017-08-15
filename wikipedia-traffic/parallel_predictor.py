import abc
import pathos.pools as pp
import numpy as np
import pandas as pd


def parallel_sub_df_col_wise_apply(df, func, processes, concat_axis=1, debug=False):
    sub_df_columns = [subset for subset in np.array_split(df.columns, processes) if len(subset) > 0]
    if debug:
        processed_sub_dfs = map(func, [df[column_subset] for column_subset in sub_df_columns])
    else:
        process_pool = pp.ProcessPool(nodes=processes)
        processed_sub_dfs = process_pool.map(func, [df[column_subset] for column_subset in sub_df_columns])
    return pd.concat(processed_sub_dfs, axis=concat_axis)


class ParallelPredictor(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, disable_parallelism=False, processes=8):
        self.disable_parallelism = disable_parallelism
        self.processes = processes

    def predict(self, X):
        if self.disable_parallelism:
            return self._serial_predict(X)
        else:
            return parallel_sub_df_col_wise_apply(X, self._serial_predict, 8)

    @abc.abstractmethod
    def _serial_predict(self, X):
        raise NotImplementedError()
