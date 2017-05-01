import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from data import X, y
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


results = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Feature selection
    svc = LinearSVC(C=0.1, dual=False)
    svc.fit(X_train, y_train)
    feature_selector = SelectFromModel(svc, prefit=True, threshold=0.2)
    X_train_reduced = feature_selector.transform(X_train)
    X_test_reduced = feature_selector.transform(X_test)

    # Classifier training
    # ~ 0.805
    #clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    # ~ 0.80
    #clf = MLPClassifier(alpha=1e-4, hidden_layer_sizes=(27,))
    # ~ 0.82
    clf = SVC(C=750, gamma=0.00025)
    clf.fit(X_train_reduced, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test_reduced))
    results.append(accuracy)

s = pd.Series(results)
print "{} +- {}".format(s.mean(), s.std() * 1.5)