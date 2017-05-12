import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from data import X, y, feature_ix_to_name, df
from debug import plot_learning_curves
import numpy as np
from matplotlib import pyplot as plt


TOP_FEATURE_COUNT = 30

# figure out which features are important
feature_importance_model = LinearSVC(C=0.1)
feature_importance_model.fit(X, y)

# Show the most important features
feature_importance = pd.Series(np.abs(feature_importance_model.coef_).sum(axis=0))
feature_importance.sort()
most_important_features = [(feature_ix_to_name(ix), importance) for ix, importance in list(feature_importance.tail(TOP_FEATURE_COUNT).iteritems())]
importance_plot = pd.DataFrame(most_important_features).set_index(0).plot(kind='bar', legend=False)
importance_plot.set_title('Top {} Most Important Features'.format(TOP_FEATURE_COUNT))
importance_plot.set_ylabel('Feature Weight')
importance_plot.set_xlabel('Feature')
plt.tight_layout()
plt.show()

# Plot a 2D graph of the structure of the dataset
essential_features = SelectFromModel(feature_importance_model, threshold=0.5, prefit=True).transform(X).todense()
pca_df = pd.concat([pd.DataFrame(PCA(n_components=2).fit_transform(essential_features)), df.category], axis=1)
distinct_colors = ['#e50000', '#736256', '#fff2bf', '#00b35f', '#397ee6', '#7f53a6', '#401023', '#f27979', '#663600', '#aab32d', '#40ffd9', '#202d40', '#4f4359', '#ff80b3', '#401100', '#e5b073', '#607339', '#59b3a1', '#002999', '#3d004d', '#994d6b', '#8c5946']
pca_plot = pca_df.plot(kind='scatter', x=0, y=1, c=np.array([distinct_colors[i] for i in y]), title='PCA Analysis of the Dataset')
pca_plot.axis('off')
plt.show()

# Plot some learning graphs
feature_selection_step = ('select_features', SelectFromModel(LinearSVC()))

# NOTE: hyper-parameter tuning doesn't happen because every parameter has a single value specified at the moment.
# This is to speed up the plotting process. For a more comprehensive result, modify the hyper-parameter ranges.
models_with_params = {
    'Random Trees': (ExtraTreesClassifier(), {'n_estimators': [50]}),
    'SVC RBF': (SVC(), {'C': [1000.0]}),
    'Neural Net': (MLPClassifier(), {'alpha': [0.0001], 'hidden_layer_sizes': [(25,)]}),
    'Naive Bayes': (MultinomialNB(), {'alpha': [1.0]})
}

pipelines = {
    name: GridSearchCV(
        Pipeline([feature_selection_step, ('clf', raw_model)]),
        param_grid=dict([('select_features__threshold', [0.2])] + [('clf__' + k, v) for k, v in params.items()]),
        verbose=10)
    for name, (raw_model, params) in models_with_params.items()
}

learning_curve_plot = plot_learning_curves(pipelines, "Learning Curves for the Craigslist Category Dataset", X, y, n_jobs=-1)
learning_curve_plot.show()
