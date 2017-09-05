import keras
from keras.constraints import maxnorm
from keras.losses import mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras import backend as K
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


def keras_smape(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(y_true + y_pred, K.epsilon(), None))
    return 200. * K.mean(diff, axis=-1)


def create_model(feature_count, layers_shape, batch_norm, learning_rate, nesterov=False, decay=0.0, reg_lambda=0.0, regularizer_str="L1", drop_out_levels=0, drop_out_prob=0.0, max_norm=None):
    model = Sequential()

    if reg_lambda == 0.:
        regularizer = None
    elif regularizer_str == "L1":
        regularizer = keras.regularizers.l1(reg_lambda)
    elif regularizer_str == "L2":
        regularizer = keras.regularizers.l2(reg_lambda)
    else:
        raise RuntimeError("Invalid regularizer specified {}".format(regularizer_str))

    if drop_out_levels > 0 and max_norm is not None:
        kernel_constraint = maxnorm(max_norm)
    else:
        kernel_constraint = None

    hidden_layers = []
    for i, next_layer_size in enumerate(layers_shape):
        input_shape = {} if i == 0 and (drop_out_levels == 0 or drop_out_prob == 0.0) else dict()
        hidden_layers.append(Dense(next_layer_size, kernel_regularizer=regularizer, kernel_constraint=kernel_constraint, input_dim=feature_count, **input_shape))

    for i, layer in enumerate(hidden_layers):
        model.add(layer)
        if batch_norm:
            model.add(BatchNormalization())
        if drop_out_levels > i and drop_out_prob > 0.0:
            model.add(Dropout(drop_out_prob))
        model.add(Activation("relu"))

    # Output layer
    model.add(Dense(1, kernel_regularizer=regularizer))

    print "layers: {}".format(model.layers)
    if nesterov:
        optimizer = keras.optimizers.Nadam(lr=learning_rate, schedule_decay=decay)
    else:
        optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay)

    model.compile(optimizer=optimizer, loss=keras_smape)
    return model


class KerasPredictor(object):
    def __init__(self, layers_shape, batch_norm=False, learning_rate=0.001, nesterov=False, decay=0.0, reg_lambda=0.0, reg_type="L1", drop_out_levels=0, drop_out_prob=0.0, max_norm=None):
        self.layers_shape = layers_shape
        self.x_scaler = RobustScaler()
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.decay = decay
        self.nesterov = nesterov
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.drop_out_levels = drop_out_levels
        self.drop_out_prob = drop_out_prob
        self.max_norm = max_norm

    def fit(self, X, y, **kwargs):
        X_normalized = self.x_scaler.fit_transform(X)
        self.model = create_model(X.shape[1],
                                  self.layers_shape,
                                  batch_norm=self.batch_norm,
                                  learning_rate=self.learning_rate,
                                  nesterov=self.nesterov,
                                  decay=self.decay,
                                  reg_lambda=self.reg_lambda,
                                  regularizer_str=self.reg_type,
                                  drop_out_prob=self.drop_out_prob,
                                  drop_out_levels=self.drop_out_levels,
                                  max_norm=self.max_norm)
        history = self.model.fit(X_normalized, y, **kwargs)
        return history

    def predict(self, X):
        X_normalized = self.x_scaler.transform(X)
        return self.model.predict(X_normalized)
