import numpy
import pandas
import pandas as pd
from keras.losses import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion

from smape import np_smape
from categorical_one_hot import CategoricalOneHotEncoder
from data import train_df
from feature_extractor import FeatureExtractor
from sklearn_pandas import DataFrameMapper
import numpy as np


def keras_smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true + y_pred), K.epsilon(), None))
    return 200. * K.mean(diff, axis=-1)


def simple_model(feature_count):
    model = Sequential()
    model.add(Dense(feature_count, input_dim=feature_count, activation="relu"))
    model.add(Dense(feature_count / 2, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))
    # TODO: model.compile(optimizer="adam", loss=keras_smape)
    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    return model


class KerasPredictor(object):
    def __init__(self, feature_count):
        self.regeressor = KerasRegressor(build_fn=lambda: simple_model(feature_count), verbose=10)

    def fit(self, X, y, **kwargs):
        self.regeressor.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.regeressor.predict(X)


sample = np.log(train_df.sample(10000, axis=1) + 1)
last_date = sample.index.values[-1]


def features_for(data, asof):
    features = FeatureExtractor(60, asof).extract_features(data)
    page_label = "page"
    date_label = "date"
    y_label = "traffic"
    indexless_features = features.reset_index().rename(columns={"index": page_label})
    flat_data = data.reset_index().rename(columns={"index": date_label}).melt(id_vars=[date_label], var_name=page_label, value_name=y_label)
    merged = pd.merge(flat_data, indexless_features, on=[date_label, page_label], how="inner")
    X = merged[list(set(merged.columns) - {date_label, page_label, y_label})]
    y = merged[y_label]
    return X, y


X_train, y_train = features_for(sample, last_date - pd.Timedelta(days=120))
X_test, y_test = features_for(sample, last_date - pd.Timedelta(days=60))

categorical_features = X_train.dtypes[X_train.dtypes.apply(lambda x: x.kind) == "O"].index.values
numerical_features = list(set(X_train.columns) - set(categorical_features))

x_transform = DataFrameMapper([(num_feature, StandardScaler()) for num_feature in numerical_features] +
                              [(cat_feature, CategoricalOneHotEncoder()) for cat_feature in categorical_features],
                              df_out=True)
y_transform = StandardScaler()

X_train_transformed = x_transform.fit_transform(X_train)
y_train_transformed = y_transform.fit_transform(y_train)

X_test_transformed = x_transform.transform(X_test)

regressor = KerasPredictor(X_train_transformed.shape[1])
regressor.fit(X_train_transformed.values, y_train_transformed)

predictions = y_transform.inverse_transform(regressor.predict(X_test_transformed.values))
print "sMAPE = {}".format(np_smape(np.exp(predictions) - 1, np.exp(y_test) - 1))

i = 0