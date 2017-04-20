from data import X, y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameters = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['hinge', 'log']
}

clf = GridSearchCV(SGDClassifier(), param_grid=parameters, n_jobs=-1)
best_model = clf.fit(X_train_scaled, y_train)
print classification_report(y_test, best_model.predict(X_test))
