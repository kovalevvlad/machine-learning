from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from data import feature_extractor, df_train, y_train, df_test
import pandas as pd


def save_predictions(predictor, X_train_, y_train_, X_test_, test_ids, output_filename):
    pipeline = Pipeline([
        ('features', feature_extractor),
        ('predictor', predictor)
    ])

    pipeline.fit(X_train_, y_train_)
    predictions = pipeline.predict(X_test_)
    submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
    submission.to_csv(output_filename, index=False)


save_predictions(ExtraTreesClassifier(n_estimators=1000), df_train, y_train, df_test, df_test.PassengerId, "trees.csv")