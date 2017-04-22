import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from data import X, y
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from debug import plot_learning_curve
from glob import glob

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# All of the below have been optimized via GridSearchCV
clfs = {
    'SGD': SGDClassifier(alpha=1e-9, loss='log', n_jobs=-1),
    'SVC': LinearSVC(C=10),
    'SVC RBF': SVC(C=100, gamma=0.01),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1),
    'Neural Net': MLPClassifier(alpha=0.001,
                                hidden_layer_sizes=(15, 10, 5),
                                solver='lbfgs',
                                activation='logistic'),
    'Radius KNN': RadiusNeighborsClassifier(radius=1.0, weights='distance', outlier_label=1),
    'Random Forrest': RandomForestClassifier(n_estimators=500, n_jobs=-1)
}

fitted_clfs = {name: clf.fit(X_train_scaled, y_train) for name, clf in clfs.items()}

# removing existing learning curves
for curve_file in glob("learning-curves/*"):
    os.remove(curve_file)

# regenerating learning curves
for name, clf in fitted_clfs.items():
    learning_curve_plot = plot_learning_curve(clf, "{} Learning Curve".format(name), X_train_scaled, y_train, scoring='f1', n_jobs=-1)
    learning_curve_plot.savefig("learning-curves/{}.png".format(name.lower().replace(" ", "-")))

# printing F1s for all classifiers
f1s = [(name, f1_score(y_test, clf.predict(X_test_scaled))) for name, clf in fitted_clfs.items()]
for name, f1 in sorted(f1s, key=(lambda name_f1: name_f1[1])):
    print "{}: {}".format(name, f1)
