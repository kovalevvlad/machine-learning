from keras.losses import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def keras_smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true + y_pred), K.epsilon(), None))
    return 200. * K.mean(diff, axis=-1)


def simple_model(feature_count, layers_shape):
    model = Sequential()
    model.add(Dense(feature_count, input_dim=feature_count, activation="relu"))
    for next_layer_size in layers_shape:
        model.add(Dense(next_layer_size, activation="relu"))
    model.add(Dense(1))
    print "layers: {}".format(model.layers)
    # TODO: model.compile(optimizer="adam", loss=keras_smape)
    model.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    return model


class KerasPredictor(object):
    def __init__(self, layers_shape):
        print "inst a nn with {}".format(layers_shape)
        self.layers_shape = layers_shape

    def fit(self, X, y, **kwargs):
        self.regeressor = Pipeline([("feature_scaling", StandardScaler()), ("predictor", KerasRegressor(build_fn=lambda: simple_model(X.shape[1], self.layers_shape), verbose=10))])
        self.regeressor.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.regeressor.predict(X)
