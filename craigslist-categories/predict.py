from collections import defaultdict

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt

from data import X, y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np


# for i in range(4):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     clf = MLPClassifier(hidden_layer_sizes=(25, 5), alpha=1e-4)
#     clf.fit(X_train, y_train)
#     f1 = f1_score(y_test, clf.predict(X_test), average="weighted")
#     print f1


results = []
for i in range(10):
    reduced_X = SelectKBest(chi2, k=15250).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.2)
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    results.append(accuracy)


s = pd.Series(results)
# 0.810484668645 +- 0.0103167074383
print "{} +- {}".format(s.mean(), s.std() * 2)
