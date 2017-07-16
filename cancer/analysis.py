import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

from data import test_df, train_df, train_y
from feature_extractor import feature_extractor

feature_extractor.fit(pd.concat([train_df, test_df]))
train_features = feature_extractor.transform(train_df)
test_features = feature_extractor.transform(test_df)

estimator = GridSearchCV(SVC(), param_grid={"C": 10. ** np.arange(-3, 4), "gamma": 10. ** np.arange(-5, 0)}, n_jobs=-1, verbose=10, scoring="accuracy")
estimator.fit(train_features, train_y)
print estimator.best_score_
print estimator.best_params_
print estimator.cv_results_
