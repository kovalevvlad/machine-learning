from collections import defaultdict

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
import pandas as pd

from data import X, y
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score


# for i in range(4):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     clf = MLPClassifier(hidden_layer_sizes=(25, 5), alpha=1e-4)
#     clf.fit(X_train, y_train)
#     f1 = f1_score(y_test, clf.predict(X_test), average="weighted")
#     print f1


for i in range(10):
    reduced_X = SelectKBest(chi2, k=14000).fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(reduced_X, y, test_size=0.2)
    clf = MultinomialNB(alpha=0.05)
    clf.fit(X_train, y_train)
    f1 = f1_score(y_test, clf.predict(X_test), average="weighted")
    print f1
