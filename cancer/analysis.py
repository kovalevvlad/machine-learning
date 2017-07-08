import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

from data import train_df, train_y, test_df
from feature_extractor import feature_extractor


feature_extractor.fit(pd.concat([train_df]))#, test_df]))
train_features = feature_extractor.transform(train_df)
# test_features = feature_extractor.transform(test_df)

estimator = LGBMClassifier()
cv_scores = pd.Series(cross_val_score(estimator, train_features, train_y.values, scoring="neg_log_loss", cv=4, n_jobs=-1, verbose=5))
print "{:.2f} +- {:.2f}".format(cv_scores.mean(), 2 * cv_scores.std())
