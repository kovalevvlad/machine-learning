import numpy as np
from sklearn.metrics import make_scorer
from data.data import y_transform


def root_mean_square_error(normalized_house_price, normalized_house_price_prediction):
    # Unscaling the values
    predictions = y_transform.inverse_transform(normalized_house_price_prediction.reshape(-1, 1))[:, 0]
    ground_truth = y_transform.inverse_transform(normalized_house_price.reshape(-1, 1))[:, 0]
    # Actual error calculation
    s = ((np.log(np.abs(predictions) + 1) - np.log(ground_truth + 1)) ** 2.0).sum()
    return np.sqrt(s / float(len(predictions)))

rmse_scorer = make_scorer(root_mean_square_error, greater_is_better=False)
