import pkg_resources
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.stats.stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import CategoricalImputer
from sklearn_pandas import DataFrameMapper


def df_from_array(X, columns):
    df = pd.DataFrame(X, columns=columns)
    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def read_csv(file_name):
    file_path = pkg_resources.resource_filename(__name__, file_name)
    return pd.read_csv(file_path)


def title(name, gender):
    title_array = pd.Series(name).str.extract("[^,]+, ([^.]+).", expand=False)
    title_remapping = {
        'Mlle': 'Miss',
        'Lady': 'Mrs',
        'the Countess': 'Mrs',
        'Ms': 'Mrs',
        'Mme': 'Mrs',
    }
    for ix, (t, g) in enumerate(zip(title_array, gender)):
        if t in title_remapping:
            title_array[ix] = title_remapping[t]
        elif t not in ('Mr', 'Miss', 'Mrs', 'Master'):
            assert g in ('male', 'female')
            title_array[ix] = 'Mr' if g == 'male' else 'Miss'
    return title_array


class ZeroDroppingTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None, copy=None):
        X_copy = X.copy()
        X_copy[X_copy == 0] = np.nan
        return X_copy

    def fit(self, X, y=None):
        return self


class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_age_by_title = None

    def transform(self, X, y=None, copy=None):
        assert X.shape[1] == 3
        title_ = title(X[:, 1], X[:, 2])
        age = X[:, 0].astype(np.float64)
        nan_mask = np.isnan(age)
        age[nan_mask] = np.array([self.mean_age_by_title[t] for t in title_[nan_mask]])
        return age.reshape((-1, 1)).astype(np.float64)

    def fit(self, X, y=None):
        assert X.shape[1] == 3
        title_ = title(X[:, 1], X[:, 2])
        age_title_df = pd.DataFrame({'age': X[:, 0].astype(np.float32), 'title': title_})
        self.mean_age_by_title = age_title_df.groupby('title').age.median().to_dict()
        return self


class CabinFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cabin_letter_encoder = CountVectorizer(binary=True, analyzer='char', vocabulary=['A', 'B', 'C', 'D', 'E', 'F', 'T'], lowercase=False)
        self.classes_ = None
        self.cabin_number_scaler = StandardScaler()

    def transform(self, X, y=None, copy=None):
        has_cabin = pd.Series(X[:, 0]).isnull()
        cabin_letters = pd.Series(X[:, 0]).fillna("").str.replace("[^A-Z]", "").values
        one_hot_cabin_letters = pd.DataFrame(self.cabin_letter_encoder.transform(cabin_letters).todense())
        cabin_number = pd.Series(X[:, 0]).str.extract("[A-Z]([0-9]+)", expand=False).astype(np.float32).fillna(0.0)
        cabin_number_scaled = pd.Series(self.cabin_number_scaler.transform(cabin_number))
        self.classes_ = ['present'] + ['letter_' + x for x in self.cabin_letter_encoder.vocabulary] + ['number']
        return pd.concat((has_cabin, one_hot_cabin_letters, cabin_number_scaled), axis=1).values.astype(np.float64)

    def fit(self, X, y=None):
        cabin_number = pd.Series(X[:, 0]).str.extract("[A-Z]([0-9]+)", expand=False).astype(np.float32).fillna(0.0)
        self.cabin_number_scaler.fit(cabin_number.values.reshape(-1, 1))

    @staticmethod
    def title(X):
        name = X[:, 1]
        gender = X[:, 2]
        title = pd.Series(name).str.extract("[^,]+, ([^.]+).")
        title_remapping = {
            'Mlle': 'Miss',
            'Lady': 'Mrs',
            'the Countess': 'Mrs',
            'Ms': 'Mrs',
            'Mme': 'Mrs',
        }
        for ix, (t, g) in enumerate(zip(title, gender)):
            if t in title_remapping:
                title[ix] = title_remapping[t]
            elif t not in ('Mr', 'Miss', 'Mrs', 'Master'):
                assert g in ('male', 'female')
                title[ix] = 'Mr' if g == 'male' else 'Miss'


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


class TitleFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.title_encoder = LabelBinarizer()
        self.classes_ = None

    def transform(self, X, y=None, copy=None):
        titles = title(X[:, 0], X[:, 1])
        return self.title_encoder.transform(titles)

    def fit(self, X, y=None):
        titles = title(X[:, 0], X[:, 1]).reshape(-1, 1)
        self.title_encoder.fit(titles)
        self.classes_ = ['Title_' + x for x in self.title_encoder.classes_]
        return self


df_train = read_csv("train.csv")
y_train = df_train.Survived
df_test = read_csv("test.csv")

feature_extractor = DataFrameMapper([
    (['Pclass'], OneHotEncoder()),
    (['Sex'], LabelEncoder()),
    (['Age', 'Name', 'Sex'], [AgeImputer(), StandardScaler()]),
    (['SibSp'], None),
    (['Parch'], None),
    (['Fare'], [ZeroDroppingTransform(), Imputer(), LogTransform(), StandardScaler()]),
    (['Cabin'], CabinFeatureExtractor()),
    (['Name', 'Sex'], TitleFeatureExtractor()),
    (['Embarked'], [CategoricalImputer(), MultiLabelBinarizer()])
])
