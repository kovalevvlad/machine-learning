from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from data import X, y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {'scoring': 'f1', 'n_jobs': -1}

clfs = {
    'sgd': GridSearchCV(
        SGDClassifier(),
        param_grid={'alpha': 10.0 ** -np.arange(1, 12), 'loss': ['hinge', 'log']},
        **params),

    'svc': GridSearchCV(LinearSVC(), param_grid={'C': 10.0 ** np.arange(-2, 10)}, **params)
}

fitted_clfs = {name: clf.fit(X_train_scaled, y_train) for name, clf in clfs.items()}

for name, fitted_clf in fitted_clfs.items():
    print "{} stats".format(name)
    print "params: {}".format(fitted_clf.best_params_)
    print classification_report(y_test, fitted_clf.predict(X_test_scaled))
    print ""