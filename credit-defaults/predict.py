from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from data import X, y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
from debug import plot_learning_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

params = {'scoring': 'f1', 'n_jobs': -1}

clfs = {
    'sgd': SGDClassifier(alpha=1e-9, loss='log'),
    'svc': LinearSVC(C=10),
    'svc rbf': SVC(C=100, gamma=0.01),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance')
}

fitted_clfs = {name: clf.fit(X_train_scaled, y_train) for name, clf in clfs.items()}

for name, fitted_clf in fitted_clfs.items():
    print "{} stats".format(name)
    print "Test Report"
    print classification_report(y_test, fitted_clf.predict(X_test_scaled))
    print "Training Report"
    print classification_report(y_train, fitted_clf.predict(X_train_scaled))
    plot_learning_curve(fitted_clf, name, X_train_scaled, y_train, **params).show()
