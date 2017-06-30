from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data import train_df, train_y
from feature_extractor import feature_extractor


pipeline = Pipeline([("features", feature_extractor), ("estimator", MultinomialNB())])
search = GridSearchCV(pipeline, param_grid={"estimator__alpha": [0.25, 0.5, 0.75, 1.]}, scoring='neg_log_loss', n_jobs=-1)
search.fit(train_df.head(1000), train_y.head(1000).values)
print search.best_score_
print search.best_params_
