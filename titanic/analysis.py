from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import plot
from data import df_train, y_train, feature_extractor


def grid_search(predictor, pipeline, param_grid, non_predictor_params=dict()):
    new_param_grid = {'predictor__' + k: v for k, v in param_grid.items()}
    new_param_grid['predictor'] = [predictor]
    combined = dict(non_predictor_params.items() + new_param_grid.items())
    return GridSearchCV(pipeline, param_grid=combined, n_jobs=-1, verbose=5, scoring='accuracy')


predictor_pipeline = Pipeline([
    ('features', feature_extractor),
    ('predictor', None)
])

clfs = {
    'Linear SVC': grid_search(LinearSVC(dual=False), predictor_pipeline, param_grid={'C': 10.0 ** np.arange(-3, 5)}),
    'KNN': grid_search(KNeighborsClassifier(), predictor_pipeline, param_grid={'n_neighbors': [3, 5, 8, 12], 'weights': ['uniform', 'distance']}),
    'Random Trees': grid_search(RandomForestClassifier(), predictor_pipeline, param_grid={'n_estimators': [10, 25, 50, 100]}),
    'Gradient Boosting': grid_search(GradientBoostingClassifier(), predictor_pipeline, param_grid={'max_depth': [2, 3, 4, 5], 'n_estimators': [25, 50, 100, 250, 500]}),
    'Extra Random Trees': grid_search(ExtraTreesClassifier(), predictor_pipeline, param_grid={'n_estimators': [10, 25, 50, 100]}),
    'RBF SVC': grid_search(SVC(), predictor_pipeline, param_grid={'C': 10.0 ** np.arange(1, 6), 'gamma': 10.0 ** np.arange(-6, 0)}),
    'Neural Net': grid_search(MLPClassifier(solver='lbfgs'), predictor_pipeline, param_grid={'alpha': 10. ** np.arange(-6, -1), 'hidden_layer_sizes': [(100,), (75, 25), (70, 20, 10), (60, 25, 10, 5)]})
}
plot.score_heatmaps(clfs, "Predictor Performance Comparison", df_train, y_train, features_excluded_from_plot=set(['predictor']), rotate_x_axis_by=30.)

clfs_fine = {
    'Gradient Boosting': grid_search(GradientBoostingClassifier(), predictor_pipeline, param_grid={'max_depth': [2, 3, 4, 5], 'n_estimators': np.arange(50, 250, 25)}),
    'RBF SVC': grid_search(SVC(), predictor_pipeline, param_grid={'C': 10.0 ** np.arange(2, 5, 0.3), 'gamma': 10.0 ** np.arange(-5, -2, 0.3)}),
}
plot.score_heatmaps(clfs_fine, "Fine-Tuned Predictor Performance Comparison", df_train, y_train, features_excluded_from_plot=set(['predictor']), rotate_x_axis_by=30.)
