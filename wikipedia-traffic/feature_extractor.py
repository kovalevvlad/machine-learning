import pandas as pd
from math_util import weekly_power_spectrum
import numpy as np
from pandas_util import safe_reindex
from parallel_predictor import parallel_sub_df_col_wise_apply
from categorical_one_hot import CategoricalOneHotEncoder
from holidays import holiday_features_for_date

page_label = u"page"
date_label = u"date"
index_column_names = [page_label, date_label]
y_label = "traffic"
categorical_feature_names = ["project", "access", "agent"]
median_periods = (14, 28, 42, 56, 112)
trend_periods = [28, 56, 112]
zero_periods = [7, 28, 56, 112]
volatility_periods = [28, 56, 112]
periodicity_periods = [56, 112]


def smooth(X, period):
    assert period % 7 == 0
    return X.rolling(period).median()


def trend_and_intercept(X):
    # trend (slope in linear regression) = cov(x,y) / var(x)
    int_range_df = pd.DataFrame(np.tile(np.arange(X.shape[0]), (X.shape[1], 1)).T, index=X.index, columns=X.columns)
    var_x = X.apply(lambda page_traffic: int_range_df[~page_traffic.isnull()][page_traffic.name].var())
    int_range_series = pd.Series(np.arange(len(X)), index=X.index)
    cov_xy = X.apply(lambda col: col.cov(int_range_series))
    trend = cov_xy / var_x
    intercept = X.mean() - int_range_df[~X.isnull()].mean() * trend
    return trend, intercept


def correlation(X):
    int_range_df = pd.DataFrame(np.tile(np.arange(X.shape[0]), (X.shape[1], 1)).T, index=X.index, columns=X.columns)
    return X.corrwith(int_range_df).fillna(0)


def series_to_32bit_dtypes(s):
    if s.dtype == np.int64:
        return s.astype(np.int32)
    elif s.dtype == np.float64:
        return s.astype(np.float32)
    else:
        return s


def df_to_32bit_dtypes(df):
    for c in df.columns:
        if df[c].dtype in (np.int64, np.float64):
            df[c] = series_to_32bit_dtypes(df[c])
    return df


class FeatureExtractor(object):
    def __init__(self, future_days, disable_parallelism=False, one_hot=True, smoothing=7):
        self.future_days = future_days
        self.disable_parallelism = disable_parallelism
        self.one_hot = one_hot
        self.smoothing = smoothing

    def extract_features(self, X):
        date_range_size = X.shape[0]
        assert date_range_size >= 112, "Need at least 112 days to extract meaningful features, instead got {}".format(date_range_size)
        if self.disable_parallelism:
            features = self._serial_extract_features(X)
        else:
            features = parallel_sub_df_col_wise_apply(X, lambda df: self._serial_extract_features(df), 8, concat_axis=0)

        # Has to be done here since after concatenating categorical series their type breaks (since new elements may appear)
        if self.one_hot:
            for categorical_feature in categorical_feature_names:
                encoder = CategoricalOneHotEncoder(feature_name=categorical_feature)
                encoded_feature = encoder.fit_transform(features[categorical_feature])
                features = pd.concat([features, encoded_feature], axis=1)

            features = features.drop(categorical_feature_names, axis=1)
        else:
            for categorical_feature in categorical_feature_names:
                if categorical_feature in features.columns:
                    features[categorical_feature] = features[categorical_feature].astype("category")

        return features

    def features_with_y(self, df):
        training_df = df.iloc[:-self.future_days]
        features = self.extract_features(training_df)

        y_2d = df.iloc[-self.future_days:]
        flat_y = safe_reindex(y_2d
                              .reset_index()
                              .rename(columns={"index": date_label})
                              .melt(id_vars=[date_label], var_name=page_label, value_name=y_label)
                              .set_index(index_column_names), features.index)[y_label]

        assert flat_y.isnull().sum() == 0, flat_y[flat_y.isnull()]
        assert np.isinf(flat_y).sum() == 0, flat_y[np.isinf(flat_y)]

        return features, series_to_32bit_dtypes(flat_y)

    def _serial_extract_features(self, X):
        # These features do not change with the date
        unchanging_features = pd.DataFrame(index=X.columns)
        weekday_mask = pd.Series(X.index.weekday < 5, index=X.index)
        weekend_mask = ~weekday_mask

        # Medians
        weekday_medians = dict()
        weekend_medians = dict()
        for median_period in median_periods:
            tail = X.tail(median_period)
            weekday_medians[median_period] = tail[weekday_mask.tail(median_period)].median()
            weekend_medians[median_period] = tail[weekend_mask.tail(median_period)].median()
            # Solid medians
            unchanging_features["median {}".format(median_period)] = tail.median()

        # Volatility
        weekday_volatilities = dict()
        weekend_volatilities = dict()
        for volatility_period in volatility_periods:
            tail = X.tail(volatility_period)
            weekday_volatilities[volatility_period] = tail[weekday_mask.tail(volatility_period)].std()
            weekend_volatilities[volatility_period] = tail[weekend_mask.tail(volatility_period)].std()
            unchanging_features["volatility {}".format(volatility_period)] = tail.std()

        # project/access/agent
        categorical_features = pd.DataFrame(
            list(pd.Series(unchanging_features.index).str.split("_").apply(lambda x: x[-3:]).values),
            columns=["project", "access", "agent"],
            index=unchanging_features.index.values)

        # remove filter
        unchanging_features = pd.concat([categorical_features, unchanging_features], axis=1)

        # periodicity
        for period in periodicity_periods:
            unchanging_features["weekly periodicity {}".format(period)] = weekly_power_spectrum(X, period)

        # Linear trends, so using X here
        trends = dict()
        intercepts = dict()
        for trend_period in trend_periods:
            tail = smooth(X, self.smoothing).tail(trend_period)
            unchanging_features["correlation {}".format(trend_period)] = correlation(tail)
            trend, intercept = trend_and_intercept(tail)
            trends[trend_period] = trend
            intercepts[trend_period] = intercept

        # Zero features
        for zero_period in zero_periods:
            unchanging_features["zeros in the last {} days".format(zero_period)] = (X.tail(zero_period) == 0.0).sum()

        # def aggregate_feature_by(source, feature, aggregate_column, aggregator=lambda x: x.mean()):
        #     aggregate = aggregator(source.groupby(aggregate_column)[feature])
        #     merged = pd.merge(aggregate.to_frame(), source[[aggregate_column]], left_index=True, right_on=aggregate_column)
        #     return merged[feature]
        #
        # TODO: Aggregate features
        # for feature_to_aggregate in [c for c in unchanging_features.columns if c.startswith("median ") or
        #                                                                        c.startswith("zeros in the last ") or
        #                                                                        c.startswith("volatility ") or
        #                                                                        c.startswith("weekly periodicity ") or
        #                                                                        c.startswith("correlation ") or
        #                                                                        c.startswith("trend ")]:
        #     for aggregate_by_feature in ["project", "access", "agent"]:
        #         new_feature_name = "median {} by {}".format(feature_to_aggregate, aggregate_by_feature)
        #         unchanging_features[new_feature_name] = aggregate_feature_by(unchanging_features, feature_to_aggregate, aggregate_by_feature)
        all_features = []
        first_predicted_date = X.index[-1] + pd.Timedelta(days=1)
        for n in range(self.future_days):
            feature_date = first_predicted_date + pd.Timedelta(days=n)
            features = pd.DataFrame(index=X.columns)

            features[date_label] = feature_date
            features["n"] = n
            features["day of week"] = feature_date.weekday()

            is_weekday = feature_date.weekday() < 5
            features["is weekday"] = is_weekday

            holiday_features = holiday_features_for_date(feature_date)
            for holiday_feature, value in holiday_features.items():
                features[holiday_feature] = value

            for trend_period in trend_periods:
                extrapolated_linear_trend_value = (trends[trend_period] * (n + trend_period) + intercepts[trend_period]).clip(lower=0)
                features["extrapolated trend {}".format(trend_period)] = extrapolated_linear_trend_value

            # Median features
            for median_period in median_periods:
                features["median {} for current day type".format(median_period)] = weekday_medians[median_period] if is_weekday else weekend_medians[median_period]

            all_features.append(pd.concat([features, unchanging_features], axis=1))

        feature_df = pd.concat(all_features, axis=0)
        feature_df = feature_df.reset_index().rename(columns={"index": page_label}).set_index(index_column_names)

        assert feature_df.isnull().sum().sum() == 0, feature_df.isnull().sum()[feature_df.isnull().sum() != 0]
        numeric_component = feature_df[feature_df.dtypes[feature_df.dtypes.apply(lambda x: str(x)).str.match("^(int|float).*")].index.values]
        assert np.isinf(numeric_component).sum().sum() == 0, np.isinf(numeric_component).sum()[np.isinf(numeric_component).sum() != 0]
        return df_to_32bit_dtypes(feature_df)
