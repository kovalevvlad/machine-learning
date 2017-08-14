import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Only works on vectors.

    Insanely, no equivalent exists in scikit-learn:
    https://github.com/scikit-learn/scikit-learn/issues/4920"""
    def __init__(self, allow_nulls=False, feature_name=None):
        self.classes_ = None
        self.allow_nulls = allow_nulls
        self.feature_name = feature_name

    def fit(self, X, y=None, copy=None):
        assert len(X.shape) == 1
        s = pd.Series(X)
        if not self.allow_nulls and s.isnull().any():
            raise RuntimeError("Found nulls in input when nulls are not allowed")
        prefix = "" if self.feature_name is None else self.feature_name + "="
        self._classes_raw = list(s.unique())
        self.classes_ = [prefix + str(x) for x in self._classes_raw]
        return self

    def transform(self, X, y=None):
        assert len(X.shape) == 1
        class_index = {x: i for i, x in enumerate(self._classes_raw)}

        if type(X) == pd.Series:
            index = X.index
            values = X.values
        else:
            index = pd.Int64Index(range(len(X)))
            values = X

        dummy_matrix = np.zeros((X.shape[0], len(self.classes_)))
        for i, value in enumerate(values):
            if value in self._classes_raw:
                dummy_matrix[i, class_index[value]] = 1
        return pd.DataFrame(dummy_matrix, columns=self.classes_, index=index)
