import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data import train_df, train_y
from feature_extractor import feature_extractor


pipeline = Pipeline([("features", feature_extractor), ("estimator", LGBMClassifier(max_depth=5, objective='multi_logloss'))])
cv_scores = pd.Series(cross_val_score(pipeline, train_df.head(1000), train_y.head(1000).values, scoring="neg_log_loss", cv=4, n_jobs=-1, verbose=5))
print "{:.2f} +- {:.2f}".format(cv_scores.mean(), 2 * cv_scores.std())
