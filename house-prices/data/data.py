from collections import OrderedDict
import pandas as pd
import pkg_resources
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import scipy.stats.stats as st


def df_from_array(X, columns):
    df = pd.DataFrame(X, columns=columns)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


class NamedTransformerMixin(TransformerMixin, BaseEstimator):
    def feature_names(self):
        raise RuntimeError("Method unimplemented")


class MagnitudeFeatureExtractor(NamedTransformerMixin):
    def __init__(self, original_df):
        # We need this to get the dtypes and column names
        self.original_df = original_df
        self.magnitude_feature_names = None

    def fit(self, X, y=None):
        df = df_from_array(X, self.original_df.columns)
        self.magnitude_feature_names = sorted(df.dtypes[df.dtypes != 'object'].index.values)
        return self

    def transform(self, X):
        df = df_from_array(X, self.original_df.columns)
        return df[self.magnitude_feature_names].values.astype(np.float64)

    def feature_names(self):
        return self.magnitude_feature_names


class CategoryFeatureExtractor(NamedTransformerMixin):
    def __init__(self, original_df, exclude_scalarized_features=False):
        # We need this to get the dtypes and column names
        self.original_df = original_df
        self.one_hot_category_feature_transforms = None
        self.category_feature_names = None
        self.exclude_scalarized_features = exclude_scalarized_features

    def fit(self, X, y=None):
        df = df_from_array(X, self.original_df.columns)

        if self.exclude_scalarized_features:
            df = df[[c for c in df.columns if c not in ScalarazingFeatureTransform.pseudo_categorical_features]]

        # Pandas parses 'NA' as np.nan even if the columns is a string column
        # Restoring the justice here!
        self.category_feature_names = sorted(df.dtypes[df.dtypes == 'object'].index.values)
        for string_feature in self.category_feature_names:
            df[string_feature] = df[string_feature].fillna("NA")

        category_features = df[self.category_feature_names]
        self.one_hot_category_feature_transforms = [CountVectorizer(token_pattern='.+').fit(feature)
                                                    for _, feature in category_features.iteritems()]
        return self

    def transform(self, X):
        df = df_from_array(X, self.original_df.columns)
        category_features = [transform.transform(raw_feature).todense()
                             for transform, (feature_name, raw_feature)
                             in zip(self.one_hot_category_feature_transforms, df[self.category_feature_names].fillna('NA').iteritems())]
        return np.hstack(category_features).astype(np.float64)

    def feature_names(self):
        def category_names_in_order(one_hot_transform):
            map_sorted_by_ix = sorted(one_hot_transform.vocabulary_.items(), key=lambda x: x[1])
            return zip(*map_sorted_by_ix)[0]

        one_hot_category_feature_names = [
            "{}: {}".format(f_name, f_value) for f_name, one_hot_transform in
            zip(self.category_feature_names, self.one_hot_category_feature_transforms) for f_value in
            category_names_in_order(one_hot_transform)]

        return one_hot_category_feature_names


class ScalarazingFeatureTransform(NamedTransformerMixin):
    quality_scale = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    quality_scale_with_missing = ['NA'] + quality_scale
    pseudo_categorical_features = OrderedDict([
        ('LotShape', ['Reg', 'IR1', 'IR2', 'IR3']),
        ('LandSlope', ['Gtl', 'Mod', 'Sev']),
        ('ExterQual', quality_scale),
        ('ExterCond', quality_scale),
        ('BsmtQual', quality_scale_with_missing),
        ('BsmtCond', quality_scale_with_missing),
        ('BsmtExposure', ['Gd', 'Av', 'Mn', 'No', 'NA']),
        ('BsmtFinType1', ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
        ('BsmtFinType2', ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
        ('HeatingQC', quality_scale),
        ('KitchenQual', quality_scale),
        ('Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']),
        ('FireplaceQu', quality_scale_with_missing),
        ('GarageFinish', ['NA', 'Unf', 'RFn', 'Fin']),
        ('GarageQual', quality_scale_with_missing),
        ('GarageCond', quality_scale_with_missing),
        ('PavedDrive', ['N', 'P', 'Y']),
        ('PoolQC', quality_scale_with_missing)
    ])

    def __init__(self, original_df):
        self.original_df = original_df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = df_from_array(X, self.original_df.columns)
        new_features = []
        for feature_name, rank in ScalarazingFeatureTransform.pseudo_categorical_features.items():
            if feature_name in self.original_df.columns:
                def category_to_float(category_str):
                    if category_str in rank:
                        return rank.index(category_str)
                    # Looks like NA can be present in features which do not mention NA in their definition
                    # In this case we assume missing data
                    elif category_str == 'NA' or np.isnan(category_str):
                        return np.nan
                    else:
                        raise RuntimeError("{} is not a member of {}".format(category_str, rank))

                # Adding a scalarized version of a pseudo-category feature
                new_features.append(df[feature_name].apply(category_to_float).values)
        return np.array(new_features).T.astype(np.float64)

    def feature_names(self):
        return ['scalarized ' + key for key in ScalarazingFeatureTransform.pseudo_categorical_features.keys() if key in self.original_df.columns]


class AbsenceFeatureExtractor(NamedTransformerMixin):
    # Features which represent elements of the house missing (NA features) for scalarized/magnitude features
    nanable_pseudo_categorical_features = OrderedDict([
        ('BsmtQual', 'basement'),
        ('FireplaceQu', 'fireplace'),
        ('PoolQC', 'pool'),
        ('LotFrontage', 'lot_frontage')
    ])

    def __init__(self, original_df):
        self.original_df = original_df

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = df_from_array(X, self.original_df.columns)
        new_features = []
        for source_feature_name, target_feature_name_root in AbsenceFeatureExtractor.nanable_pseudo_categorical_features.items():
            # If feature has not been deleted
            if source_feature_name in df.columns:
                new_features.append(df[source_feature_name].isnull())
        return np.array(new_features).T.astype(np.float64)

    def feature_names(self):
        return [v + " missing" for k, v in AbsenceFeatureExtractor.nanable_pseudo_categorical_features.items() if k in self.original_df.columns]


class LogTransform(BaseEstimator, TransformerMixin):
    def __init__(self, max_skew=None):
        self.max_skew = 0 if max_skew is None else max_skew
        self.transform_mask = None

    def transform(self, X, y=None, copy=None):
        assert X.dtype in {np.dtype('float64'), np.dtype('float32')}
        X_T = X.copy().T
        X_T[self.transform_mask] = np.log(X_T[self.transform_mask] + 1)
        return X_T.T

    def inverse_transform(self, X, copy=None):
        X_T = X.copy().T
        X_T[self.transform_mask] = np.exp(X_T[self.transform_mask]) - 1
        return X_T.T

    def fit(self, X, y=None):
        self.transform_mask = st.skew(X, nan_policy='omit') > self.max_skew
        return self


class MedianNeighbourhoodHousePriceFeatureExtractor(NamedTransformerMixin):
    def __init__(self, original_df):
        self.original_df = original_df
        self.median_prices = None

    def transform(self, X, y=None, copy=None):
        df = df_from_array(X, self.original_df.columns)
        return pd.merge(df[['Neighborhood']], self.median_prices, left_on='Neighborhood', right_index=True, how='left').sort_index()['p'].values.reshape(-1, 1)

    def fit(self, X, y=None):
        df = df_from_array(X, self.original_df.columns)
        neighborhood_price = pd.DataFrame({'n': df['Neighborhood'], 'p': y[:, 0]})
        self.median_prices = neighborhood_price.groupby('n').median()
        return self

    def feature_names(self):
        return ['median_neighborhood_price']


def read_csv(file_name):
    file_path = pkg_resources.resource_filename(__name__, file_name)
    df = pd.read_csv(file_path)
    return df


df_test = read_csv('test.csv')
df_train = read_csv('train.csv')

# Normalizing house prices
y_transform = Pipeline([("log_scaling", LogTransform()), ("standard_scaling", StandardScaler())])
y_train = y_transform.fit_transform(df_train.SalePrice.values.reshape(-1, 1).astype(np.float64))

del df_train['SalePrice']
del df_train['Id']

raw_feature_extraction_stage = FeatureUnion([
     ('scalarized', ScalarazingFeatureTransform(df_train)),
     ('category', CategoryFeatureExtractor(df_train, exclude_scalarized_features=False)),
     ('magnitude', Pipeline(
        [("extraction", MagnitudeFeatureExtractor(df_train)), ('impute', Imputer()),
         ("log_scale", LogTransform(0.6))])),
     ('median_neighbourhood_price', MedianNeighbourhoodHousePriceFeatureExtractor(df_train))])

predictor_pipeline = Pipeline([
    ('raw', raw_feature_extraction_stage),
    ('impute', Imputer()),
    ('drop_constant', VarianceThreshold()),
    ('scaling', RobustScaler()),
    ('feature_selection', SelectKBest(score_func=mutual_info_regression, k=230)),
    ('predictor', None)])
