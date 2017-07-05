import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class CategoricalOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Only works on vectors.

    Insanely, no equivalent exists in scikit-learn:
    https://github.com/scikit-learn/scikit-learn/issues/4920"""
    def __init__(self, allow_nulls=False):
        self.classes_ = None
        self.allow_nulls = allow_nulls

    def fit(self, X, y=None, copy=None):
        assert len(X.shape) == 1
        data = pd.DataFrame({"data": X})
        if not self.allow_nulls and data.data.isnull().any():
            raise RuntimeError("Found nulls in input when nulls are not allowed")
        self.classes_ = data.data.unique()
        return self

    def transform(self, X, y=None):
        assert len(X.shape) == 1
        data = pd.DataFrame({"data": X})
        class_index = {x: i for i, x in enumerate(self.classes_)}
        dummy_matrix = np.zeros((data.shape[0], len(self.classes_)))
        for i, value in enumerate(data.data.values):
            if value in class_index:
                dummy_matrix[i, class_index[value]] = 1
        return dummy_matrix
