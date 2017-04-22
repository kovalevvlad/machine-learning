from sklearn.ensemble import RandomForestClassifier

from data import X, y
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=100)

clf.fit(X_train_scaled, y_train)
print "Test Report"
print classification_report(y_test, clf.predict(X_test_scaled))
