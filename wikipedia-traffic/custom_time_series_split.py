import numpy as np


class CustomTimeseriesSplit:
    def __init__(self, test_slice_size, n_splits, min_days_for_training):
        # TODO: Use n_splits wiser
        self.n_splits = n_splits
        self.test_slice_size = test_slice_size
        self.min_training_samples = min_days_for_training

    def split(self, X):
        n_samples = X.shape[0]
        assert self.min_training_samples + self.n_splits + self.test_slice_size <= n_samples
        indices = np.arange(n_samples)
        for test_start_float in np.linspace(self.min_training_samples, n_samples - self.test_slice_size, num=self.n_splits):
            test_start = int(np.floor(test_start_float))
            train = indices[:test_start]
            test = indices[test_start:test_start + self.test_slice_size]
            assert test.shape[0] == self.test_slice_size
            yield (X.iloc[train], X.iloc[test])
