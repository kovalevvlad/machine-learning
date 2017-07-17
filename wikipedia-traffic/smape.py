def smape(ground_truth, predictions):
    """
    As defined at https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """

    assert ground_truth.shape == predictions.shape
    assert (ground_truth.index == predictions.index).all()
    assert (ground_truth.columns == predictions.columns).all()

    width, height = ground_truth.shape
    n = width * height

    denominator = ground_truth + predictions
    summation_parts = (ground_truth - predictions).abs() / denominator
    # As defined by the rules, whenever both ground_truth and predictions are 0 (hence denominator is 0)
    # we define smape to be 0
    for column in ground_truth.columns:
        summation_parts[column][denominator[column] == 0] = 0
    return 100 * 2 * summation_parts.sum().sum() / n
