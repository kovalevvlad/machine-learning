import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy import signal


def rolling_harmonic_mean(x, window_size):
    x_inv = 1 / x
    rolling_inv_sum = x_inv.rolling(window_size).sum()
    return window_size / rolling_inv_sum


def fuller_test(ts):
    dftest = adfuller(ts.diff().fillna(0), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput


def weekly_power_spectrum(X, tail, sliding_window=28):
    """
    Weekly power spectrum of the last `tail` elements of X. The index represents frequency in weeks.
    """
    last_values = X.tail(tail)
    # Weekly value
    spectrum = power_spectrum(last_values, sliding_window)
    return spectrum.loc[1.0]


def power_spectrum(X, sliding_window):
    assert X.isnull().sum().sum() == 0
    # de-trending
    X = X - X.rolling(14, min_periods=1).median()
    f, Pxx = signal.welch(X, nperseg=sliding_window, axis=0, fs=7.)
    # Scaling
    spectrum = pd.DataFrame(Pxx, index=f, columns=X.columns) / pd.DataFrame(Pxx, index=f, columns=X.columns).sum()
    # Setting constant value's periodicity to 0
    return spectrum.fillna(0)


class Despiker:
    def __init__(self, rolling_median_period=7, ewmstd_halflife=14):
        self.rolling_median_period = rolling_median_period
        self.ewmstd_halflife = ewmstd_halflife

    # def fit_transform(self, s):
    #     std = pd.ewmstd(s, halflife=self.ewmstd_halflife, min_periods=5)
    #     median = s.rolling(self.rolling_median_period).median()
    #     self.low = (median - std * 2).shift(1)
    #     self.high = (median + std * 2).shift(1)
    #     high_spikes = s > self.high
    #     low_spikes = s < self.low
    #     result = s.copy()
    #     result[high_spikes] = self.high[high_spikes]
    #     result[low_spikes] = self.low[low_spikes]
    #     return result
    #
    def fit_transform(self, s):
        median = s.rolling(self.rolling_median_period).median()
        sdiff = s - median
        std = pd.ewmstd(sdiff, halflife=self.ewmstd_halflife, min_periods=5)
        # Useful for debugging
        self.low = (median - std * 2).shift(1)
        self.high = (median + std * 2).shift(1)
        high_spikes = s > self.high
        low_spikes = s < self.low
        result = s.copy()
        result[high_spikes] = self.high[high_spikes]
        result[low_spikes] = self.low[low_spikes]
        return result


def assert_no_nulls(df):
    assert not df.isnull().any().any()


def assert_no_infs(df):
    assert not (df.abs() == np.inf).any().any()