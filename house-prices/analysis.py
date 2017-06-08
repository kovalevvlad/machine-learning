from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer
import seaborn as sns

import plot
from data.data import y_train, y_transform, predictor_pipeline, df_train, df_test, NamedTransformerMixin, raw_feature_extraction_stage
import matplotlib.pyplot as plt
from sklearn import manifold, svm
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from scoring import rmse_scorer


def recursive_transform_to_output_feature_names(transform):
    if isinstance(transform, FeatureUnion):
        return [feature_name for t in transform.transformer_list for feature_name in recursive_transform_to_output_feature_names(t[1])]
    elif isinstance(transform, Pipeline):
        return recursive_transform_to_output_feature_names(transform.steps[0][1])
    else:
        assert isinstance(transform, NamedTransformerMixin)
        return transform.feature_names()


# Extracting raw features using the initial data processing staged
# Imputer is necessary here since MDS cannot handle NANs
raw_feature_extraction_stage_with_imputation = Pipeline([('raw', raw_feature_extraction_stage), ('impute', Imputer())])
raw_features = raw_feature_extraction_stage_with_imputation.fit_transform(df_train.values, y_train)

# # MDS Plot
mds = manifold.MDS(n_jobs=-1, max_iter=100)
X_mds = pd.DataFrame(mds.fit_transform(raw_features), columns=('x', 'y'))
mds_plot = plt.scatter(X_mds.x, X_mds.y, c=y_train, cmap='winter')
plt.axis('off')
plt.suptitle("2D MDS Transform")
colorbar = plt.colorbar(mds_plot)
colorbar.set_label("House Price (normalized)")
plt.show()

# Normalizing house prices
y_train_unscaled = y_transform.inverse_transform(y_train)
pd.Series(y_train_unscaled[:, 0]).hist()
plt.title("House Prices Before Normalization")
plt.show()

pd.Series(y_train[:, 0]).hist()
plt.title("House Prices After Normalization")
plt.show()

# Identifying the most important features
raw_feature_names = recursive_transform_to_output_feature_names(raw_feature_extraction_stage_with_imputation)
mutual_info = pd.Series(mutual_info_regression(raw_features, y_train), index=raw_feature_names).sort_values()

TOP_FEATURE_COUNT = 30
most_important_features = mutual_info.tail(TOP_FEATURE_COUNT)
plot.features_with_weight(most_important_features, 'Top {} Features with the Most Predictive Power'.format(TOP_FEATURE_COUNT))

# Plot features sorted by mutual info
sorted_feature_weight_plot = pd.Series(data=mutual_info.values, index=reversed(range(len(mutual_info)))).plot(grid=True, title="Features Sorted by their Mutual Information with the House Price")
sorted_feature_weight_plot.set_ylabel("Feature Mutual Information with the House Price")
sorted_feature_weight_plot.set_xlabel("Feature Mutual Information Rank")
plt.show()

# Feature Correlation Analysis
corr_matrix = pd.DataFrame(raw_features, columns=raw_feature_names)[most_important_features.index.values].corr().abs()
corr_plot = sns.heatmap(corr_matrix, square=True)
corr_plot.set_xticklabels(corr_matrix.columns.values, rotation=30., horizontalalignment='right')
corr_plot.set_yticklabels(reversed(corr_matrix.index.values), rotation=0)
plt.show()

# Drop blacklisted features
blacklisted_features = [
        # correlated with GarageCars and the latter has a higher correlation coeff with the house price
        # TODO: resolve the following correlated feature pairs:
        # - Fireplaces & Fireplaces: na,
        # - ExterQual: ta & ExterQual: gd
        # - scalarized ExterQual & ExterQual: ta
        'GarageArea',
        # All of the below have too many missing values
        'MiscFeature']

for blacklisted_feature in blacklisted_features:
    del df_train[blacklisted_feature]
    del df_test[blacklisted_feature]

# Outlier analysis
for feature_name in ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']:
    df_train[feature_name].hist(bins=25)
    plt.title(feature_name)
    plt.show()

# Estimator performance analysis
grid_search_kwargs = {'n_jobs': -1, 'verbose': 5, 'scoring': rmse_scorer}
clfs = {
    'Gradient Boosting': GridSearchCV(predictor_pipeline, param_grid={'predictor': [GradientBoostingRegressor()], 'predictor__max_depth': [1, 2, 3, 4], 'predictor__loss': ['ls', 'lad', 'huber', 'quantile']}, **grid_search_kwargs),
    'Linear SVM': GridSearchCV(predictor_pipeline, param_grid={'predictor': [svm.LinearSVR()], 'predictor__C': 10.0 ** np.arange(-6, 2)}, **grid_search_kwargs),
    'RBF SVM': GridSearchCV(predictor_pipeline, param_grid={'predictor': [svm.SVR(cache_size=1000)], 'predictor__C': 10.0 ** np.arange(-1, 5), 'predictor__gamma': 10.0 ** np.arange(-5, 0)}, **grid_search_kwargs),
    'Elastic Net': GridSearchCV(predictor_pipeline, param_grid={'predictor': [ElasticNet()], 'predictor__alpha': 10.0 ** np.arange(-5, 1)}, **grid_search_kwargs),
    'Random Forrest': GridSearchCV(predictor_pipeline, param_grid={'predictor': [RandomForestRegressor()], 'predictor__n_estimators': [10, 25, 50, 100]}, **grid_search_kwargs),
    'Ada Boost': GridSearchCV(predictor_pipeline, param_grid={'predictor': [AdaBoostRegressor()], 'predictor__n_estimators': [10, 25, 50, 100]}, **grid_search_kwargs),
    'Ridge': GridSearchCV(predictor_pipeline, param_grid={'predictor': [Ridge()], 'predictor__alpha': 10.0 ** np.arange(-2, 7)}, **grid_search_kwargs),
    'Lasso': GridSearchCV(predictor_pipeline, param_grid={"predictor": [Lasso()], 'predictor__alpha': 10.0 ** np.arange(-6, 0)}, **grid_search_kwargs),
    'Extra Random Forrest': GridSearchCV(predictor_pipeline, param_grid={"predictor": [ExtraTreesRegressor()], 'predictor__n_estimators': [10, 25, 50, 100]}, **grid_search_kwargs),
    'Poly SVM': GridSearchCV(predictor_pipeline, param_grid={"predictor": [svm.SVR(kernel='poly')], 'predictor__C': 10.0 ** np.arange(-3, 4), 'predictor__gamma': 10.0 ** np.arange(-4, 1)}, **grid_search_kwargs),
    'Sigmoid SVM': GridSearchCV(predictor_pipeline, param_grid={"predictor": [svm.SVR(kernel='sigmoid')], 'predictor__C': 10.0 ** np.arange(-2, 3), 'predictor__gamma': 10.0 ** np.arange(-5, 0)}, **grid_search_kwargs),
    'Neural Net': GridSearchCV(predictor_pipeline, param_grid={"predictor": [MLPRegressor(solver='lbfgs')], 'predictor__alpha': 10.0 ** np.arange(-2, 4), 'predictor__hidden_layer_sizes': [(70,), (35, 35), (65, 5), (45, 17, 8), (38, 22, 16, 11, 8, 5)]}, **grid_search_kwargs),
}
plot.score_heatmaps(clfs, "Predictor Performance Comparison", df_train.values, y_train, features_excluded_from_plot=set(['predictor']), rotate_x_axis_by=30., clipping_score=-0.2)

# RBF SVM Fine Tuning
plot.score_heatmaps(
    {
        'RBF SVM': GridSearchCV(
            predictor_pipeline,
            param_grid={
                'predictor': [svm.SVR(cache_size=1000)],
                'predictor__C': 10.0 ** np.arange(0, 3.1, 0.3),
                'predictor__gamma': 10.0 ** np.arange(-4, -1.9, 0.3)
            },
            **grid_search_kwargs),
    },
    "RBF SVM Performance Fine Tuning",
    df_train.values,
    y_train,
    features_excluded_from_plot=set(['predictor']),
    rotate_x_axis_by=30.,
    clipping_score=-0.2)
