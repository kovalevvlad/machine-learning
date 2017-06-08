from data.data import y_train, df_test, y_transform, df_train, predictor_pipeline
import pandas as pd


def save_predictions(predictor, X_train_, y_train_, X_test_, test_ids, y_transform_, output_filename):
    predictor.fit(X_train_, y_train_)
    predictions = predictor.predict(X_test_)
    predicted_prices = y_transform_.inverse_transform(predictions.reshape(-1, 1))[:, 0]
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predicted_prices})
    submission.to_csv(output_filename, index=False)

save_predictions(
    predictor_pipeline,
    df_train.values,
    y_train,
    df_test[[c for c in df_test.columns if c != 'Id']].values,
    df_test['Id'].values,
    y_transform,
    "predictions.csv")
