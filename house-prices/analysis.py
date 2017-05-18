from sklearn.metrics import make_scorer
from sklearn.preprocessing import scale

from data.data import X_train, y_train, train_feature_names
import matplotlib.pyplot as plt
from sklearn import manifold, svm
import pandas as pd
import numpy as np


def plot_features_with_weight(features, title):
    negative_plot = (pd.DataFrame(features)[1] < 0).any()
    ylim = (-0.25, 0) if negative_plot else (0, 0.25)
    feature_plot = pd.DataFrame(features).set_index(0).plot(kind='bar', legend=False, ylim=ylim)
    feature_plot.set_title(title)
    feature_plot.set_ylabel('Feature Weight')
    feature_plot.set_xlabel('Feature')
    plt.tight_layout()
    plt.show()


TOP_FEATURE_COUNT = 30

# MDS Plot
mds = manifold.MDS(n_jobs=-1, max_iter=100)
X_mds = pd.DataFrame(mds.fit_transform(X_train), columns=('x', 'y'))

plot = plt.scatter(X_mds.x, X_mds.y, c=y_train)
plt.axis('off')
plt.suptitle("2D MDS Transform")
colorbar = plt.colorbar(plot)
colorbar.set_label("House Price")
plt.show()


# As defined by https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError
def log_mean_square_error(ground_truth, predictions):
   sum = ((np.log(predictions + 1) - np.log(ground_truth + 1)) ** 2.0).sum()
   return np.sqrt(sum / float(len(predictions)))


lms_scorer = make_scorer(log_mean_square_error, greater_is_better=False)


# Identifying the most important features
feature_importance_model = svm.LinearSVR(C=0.03)
feature_importance_model.fit(X_train, scale(y_train))

feature_importance = pd.Series(feature_importance_model.coef_).sort_values()
abs_feature_importance = feature_importance.abs().sort_values()

most_important_positive_features = [(train_feature_names[ix], importance) for ix, importance in list(feature_importance.tail(TOP_FEATURE_COUNT).iteritems())]
most_important_negative_features = [(train_feature_names[ix], importance) for ix, importance in list(feature_importance.head(TOP_FEATURE_COUNT).sort_values(ascending=False).iteritems())]
least_important_features = [(train_feature_names[ix], importance) for ix, importance in list(abs_feature_importance.head(TOP_FEATURE_COUNT).iteritems())]

plot_features_with_weight(most_important_positive_features, 'Top {} Features with the Most Positive Price Impact'.format(TOP_FEATURE_COUNT))
plot_features_with_weight(most_important_negative_features, 'Top {} Features with the Most Negative Price Impact'.format(TOP_FEATURE_COUNT))
plot_features_with_weight(least_important_features, 'Top {} Features with the Least Price Impact'.format(TOP_FEATURE_COUNT))

# Plot sorted feature weights
sorted_feature_weight_plot = pd.Series(data=abs_feature_importance.values, index=range(len(feature_importance))).plot(grid=True, title="Sorted Feature Weights")
sorted_feature_weight_plot.set_ylabel("Feature Weight")
sorted_feature_weight_plot.set_xlabel("Feature ID")
plt.show()
