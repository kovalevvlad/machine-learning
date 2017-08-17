# coding=utf-8
import datetime

import pytest

from feature_extractor import FeatureExtractor, median_periods, volatility_periods, zero_periods, trend_periods, periodicity_periods
import feature_extractor
import pandas as pd
import numpy as np

from pandas_util import safe_reindex

index = pd.date_range(start=datetime.datetime(2015, 1, 1), periods=360, name=feature_extractor.date_label)
future_days = 60
valid_page_names = [
    u'Special:MyLanguage/Help:Extension:Translate_www.mediawiki.org_mobile-web_all-agents',
    u'África_es.wikipedia.org_desktop_all-agents',
    u'Claire_Forlani_de.wikipedia.org_all-access_spider',
    u'Das_Geisterhaus_(Film)_de.wikipedia.org_desktop_all-agents',
    u'Art_Schwind_en.wikipedia.org_all-access_all-agents',
    u'Rhinolophus_hipposideros_fr.wikipedia.org_all-access_spider',
    u'List_of_stock_market_crashes_and_bear_markets_en.wikipedia.org_all-access_spider',
    u'Tolerance_tax_en.wikipedia.org_all-access_all-agents',
    u'Маркетинг_ru.wikipedia.org_mobile-web_all-agents',
    u'Crystal_Clear_commons.wikimedia.org_desktop_all-agents'
]


def generate_X(values=range(len(index)), one_column=True):
    page_selector = [valid_page_names[0]] if one_column else valid_page_names
    missing_data_point_count = len(index) - len(values)
    frame = pd.DataFrame({page_name: np.concatenate((np.zeros((missing_data_point_count,)), np.array(values))) for page_name in valid_page_names}, index=index)
    X = frame[page_selector]
    return X


def expected_future_index(X, length):
    return pd.date_range(X.index.values[-1] + pd.Timedelta(days=1), periods=length)


def test_flat_median_features(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X()
    features, norm_log_median = fe.extract_features(X)
    for median_period in median_periods:
        median_feature = features["median {}".format(median_period)].loc[X.columns[0]]
        adjusted_X = normalize_X(X, norm_log_median)
        expected_feature_value = adjusted_X.tail(median_period).median()
        assert median_feature.unique() == [expected_feature_value]


def normalize_X(X, norm_log_median):
    return np.log(X + 1) - norm_log_median


def test_n(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X()
    features, _ = fe.extract_features(X)
    future_index = expected_future_index(X, fe.future_days)
    expected_n = pd.Series(range(fe.future_days), index=future_index)
    assert (features.reset_index().set_index(feature_extractor.date_label)["n"] == expected_n).all()


def test_zeroes(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X(values=np.random.randint(0, high=2, size=120))
    features, _ = fe.extract_features(X)
    for period in zero_periods:
        expected = (X.tail(period) == 0).sum().sum()
        feature = features["zeros in the last {} days".format(period)]
        actual = feature.unique()
        assert actual == [expected]


def test_split_median(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X(values=np.random.random(len(index)))
    features, normalized_log_median = fe.extract_features(X)
    adjusted_X = normalize_X(X, normalized_log_median)
    for period in median_periods:
        # actual
        feature_name = "median {} for current day type".format(period)
        actual = features.reset_index().set_index(feature_extractor.date_label)[feature_name]

        # expected
        adjusted_X_tail = adjusted_X[adjusted_X.columns[0]].tail(period)
        weekday_mask = adjusted_X_tail.index.weekday < 5
        weekday_median = adjusted_X_tail[weekday_mask].median()
        weekend_median = adjusted_X_tail[~weekday_mask].median()

        future_index = features.reset_index()[feature_extractor.date_label]
        future_weekday = pd.Series((future_index.dt.weekday < 5).values, future_index)
        expected_weekday_series = pd.Series([weekday_median] * future_days, index=future_index)
        expected_weekday_series[~future_weekday] = 0.
        expected_weekend_series = pd.Series([weekend_median] * future_days, index=future_index)
        expected_weekend_series[future_weekday] = 0.
        expected_feature = expected_weekday_series + expected_weekend_series

        assert (actual == expected_feature).all(), "period={}".format(period)


def test_correlation(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X(values=range(180))
    features, _ = fe.extract_features(X)
    for period in trend_periods:
        feature = features["correlation {}".format(period)]
        assert list(feature[X.columns[0]].unique())[0] == pytest.approx(1., rel=0.00001)

    X = generate_X(values=np.repeat([0], 180))
    features, _ = fe.extract_features(X)
    for period in trend_periods:
        feature = features["correlation {}".format(period)]
        assert list(feature[X.columns[0]].unique())[0] == pytest.approx(0., abs=0.00001)

    X = generate_X(values=np.arange(180) + np.random.rand(180))
    features, _ = fe.extract_features(X)
    for period in trend_periods:
        feature = features["correlation {}".format(period)]
        assert list(feature[X.columns[0]].unique())[0] < 1.
        assert list(feature[X.columns[0]].unique())[0] > 0.8


def test_volatility(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X(values=np.repeat([123], 180))
    features, _ = fe.extract_features(X)
    for period in volatility_periods:
        for feature_name_template in ["volatility {}"]:
            feature_name = feature_name_template.format(period)
            feature = features[feature_name]
            assert feature.unique()[0] == pytest.approx(0., abs=0.000001)

    X = generate_X(values=np.random.rand(180))
    X1 = generate_X(values=np.random.rand(180) * 100)
    features, _ = fe.extract_features(X)
    features1, _ = fe.extract_features(X1)
    for period in volatility_periods:
        for feature_name_template in ["volatility {}"]:
            feature_name = feature_name_template.format(period)
            feature = features[feature_name]
            feature1 = features1[feature_name]
            assert feature.unique()[0] > 0.
            assert (feature < feature1).all()

    for period in volatility_periods:
        feature_name = "volatility {}".format(period)
        feature = features[feature_name]
        feature1 = features1[feature_name]
        assert feature.mean() > 0.
        assert (feature < feature1).all()
        assert len(feature.unique()) == 1


def test_periodicity(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    weekly_saw = generate_X(np.tile([1, 2, 3, 4, 5, 6, 7], 22))
    noisy_weekly_saw = weekly_saw + np.random.rand(*weekly_saw.shape) * 7
    line = generate_X()
    ws_features, _ = fe.extract_features(weekly_saw)
    noisy_ws_features, _ = fe.extract_features(noisy_weekly_saw)
    line_features, _ = fe.extract_features(line)
    for period in periodicity_periods:
        feature_name = "weekly periodicity {}".format(period)
        assert (ws_features[feature_name] > noisy_ws_features[feature_name]).all()
        assert (noisy_ws_features[feature_name] > line_features[feature_name]).all()


def test_trend(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism)
    X = generate_X()
    expected = X.iloc[-60:]

    X = X.iloc[:-60]
    X.iloc[-1] = 1e9
    X.iloc[-5] = 1e9

    expected = expected[expected.columns[0]]
    features, normalizing_log_median = fe.extract_features(X)

    for period in trend_periods:
        feature_name = "extrapolated trend {}".format(period)
        normalized_feature = features[feature_name]
        feature = np.exp(normalized_feature + normalizing_log_median.values[0]) - 1
        feature = feature.reset_index().drop(labels=[feature_extractor.page_label], axis=1).set_index([feature_extractor.date_label])[feature_name]
        expected_lower_bound = expected * 0.99
        expected_upper_bound = expected * 1.01
        assert (expected_lower_bound < feature).all()
        assert (expected_upper_bound > feature).all()


def test_project_acces_agent(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism, one_hot=False)
    X = generate_X(one_column=False)
    features, _ = fe.extract_features(X)
    page_info = {p: p.split("_")[-3:] for p in valid_page_names}

    for feature_name, feature_value_index in [("project", 0), ("access", 1), ("agent", 2)]:
        for page in valid_page_names:
            assert (features.loc[page, :][feature_name] == page_info[page][feature_value_index]).all()

    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism, one_hot=True)
    features, _ = fe.extract_features(X)

    for feature_name, feature_value_index in [("project", 0), ("access", 1), ("agent", 2)]:
        for page in valid_page_names:
            feature_columns = [x for x in features.columns if x.startswith(feature_name + "=")]
            for feature_column in feature_columns:
                assert (features.loc[page, :][feature_column] ==
                        (1. if page_info[page][feature_value_index] in feature_column else 0.)).all(), "feature: {}, page: {}".format(feature_name, page)


def test_extract_with_y(disable_parallelism):
    fe = FeatureExtractor(future_days, disable_parallelism=disable_parallelism, one_hot=True)
    X = generate_X(one_column=False)
    features, normalizing_log_median = fe.extract_features(X.iloc[:-future_days])
    features_via_with_y, normalized_y, normalizing_log_median_via_with_y = fe.features_with_y(X)

    assert (features == features_via_with_y).all().all()
    assert (normalizing_log_median == normalizing_log_median_via_with_y).all()

    raw_y = X.iloc[-future_days:]
    norm_raw_y = np.log(raw_y + 1) - normalizing_log_median
    expected_y = (norm_raw_y
                  .reset_index()
                  .melt(id_vars=[feature_extractor.date_label], var_name=feature_extractor.page_label, value_name=feature_extractor.y_label)
                  .set_index(feature_extractor.index_column_names)[feature_extractor.y_label])
    expected_y = safe_reindex(expected_y, normalized_y.index)
    assert (normalized_y == expected_y).all()
