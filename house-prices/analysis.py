from matplotlib import cm
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import scale
from sklearn.svm import NuSVR

from data.data import X_train, y_train, train_feature_names, y_train_scaled, y_scaler
import matplotlib.pyplot as plt
from sklearn import manifold, svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pandas as pd
import numpy as np


def plot_features_with_weight(features, title):
    negative_plot = (pd.DataFrame(features)[1] < 0).any()
    ylim = (-0.25, 0) if negative_plot else (0, 0.25)
    feature_plot = pd.DataFrame(features).set_index(0).plot(kind='bar', legend=False, ylim=ylim)
    feature_plot.set_title(title)
    feature_plot.set_ylabel('Feature Weight')
    feature_plot.set_xlabel('Feature')
    feature_plot.tight_layout()
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

# Identifying the most important features
# TODO: Rotate graphs to make the feature names more readable.
feature_importance_model = svm.LinearSVR(C=0.03)
feature_importance_model.fit(X_train, y_train_scaled)

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


# Analyse estimator performance

# As defined by https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError
def log_mean_square_error(ground_truth_scaled, predictions_scaled):
    # Unscaling the values
    predictions = y_scaler.inverse_transform(predictions_scaled)
    ground_truth = y_scaler.inverse_transform(ground_truth_scaled)
    # Actual error calculation
    s = ((np.log(np.abs(predictions) + 1) - np.log(ground_truth + 1)) ** 2.0).sum()
    return np.sqrt(s / float(len(predictions)))


lms_scorer = make_scorer(log_mean_square_error, greater_is_better=False)


def plot_score_heatmaps(grid_searches):
    subplot_count = len(grid_searches)
    n_columns = int(np.ceil(np.sqrt(float(subplot_count))))
    n_rows = subplot_count / n_columns + (0 if subplot_count % n_columns == 0 else 1)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False)

    for ax, (estimator_name, grid_search) in zip(axes.flat, grid_searches.items()):
        print "Fitting {}".format(estimator_name)
        grid_search.fit(X_train, y_train_scaled)
        df = pd.DataFrame([dict(param_dict.items() + [('score', score)]) for score, param_dict in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params'])])
        params = list(set(df.columns) - {'score'})
        assert len(params) in (1, 2)
        if len(params) == 1:
            param_name = params[0]
            im_data = np.array([df.score]).T
            cax = ax.imshow(im_data, vmin=df.score.min(), vmax=df.score.max(), aspect='auto', extent=(0, 1, 0, len(df.score)), origin='lower', cmap='plasma')
            best_score_ix = im_data.argmax()
            ax.text(0.5, best_score_ix + 0.5, '{0:.3f}'.format(float(im_data[best_score_ix])))
            ax.set_ylabel(param_name)
            # for some reason matplotlib refuses to display the first tick... shift it along!
            ticks = df[param_name].values
            ax.set_yticklabels(ticks)
            ax.set_yticks(0.5 + np.arange(len(ticks)))
            ax.axes.get_xaxis().set_visible(False)
        if len(params) == 2:
            ix_param = params[0]
            col_param = params[1]
            pivoted_df = df.pivot(index=ix_param, columns=col_param)
            im_data = pivoted_df.values
            cax = ax.imshow(im_data, vmin=df.score.min(), vmax=df.score.max(), aspect='auto', extent=(0, len(pivoted_df.columns.levels[1]), 0, len(pivoted_df.index)), origin='lower', cmap='plasma')
            best_score_x, best_score_y = np.unravel_index(im_data.argmax(), im_data.shape)
            ax.text(best_score_y + 0.1, best_score_x + 0.5, '{0:.3f}'.format(float(im_data[best_score_x, best_score_y])))
            ax.set_ylabel(ix_param)
            ax.set_xlabel(col_param)
            # for some reason matplotlib refuses to display the first tick... shift it along!
            yticks = pivoted_df.index.values
            ax.set_yticklabels(yticks)
            ax.set_yticks(0.5 + np.arange(len(yticks)))
            xticks = pivoted_df.columns.levels[1]
            has_long_labels = xticks.dtype.name == 'object' and max([len(str(tick)) for tick in xticks]) > 5
            rotation = 15. if has_long_labels else 'horizontal'
            alignment = 'right' if has_long_labels else 'center'
            ax.set_xticklabels(xticks, rotation=rotation, horizontalalignment=alignment)
            ax.set_xticks(0.5 + np.arange(len(xticks)))

        ax.set_title(estimator_name)
        colorbar = plt.colorbar(cax, ax=ax)
        colorbar.set_label("Cross-Validation Score")

    hidden_axes = axes.flat[len(grid_searches):]
    for hidden_axis in hidden_axes:
        hidden_axis.set_visible(False)

    plt.suptitle('Estimator Performance')
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()


grid_search_kwargs = {'n_jobs': -1, 'verbose': 2, 'scoring': lms_scorer}
clfs = {
    'Linear SVM': GridSearchCV(svm.LinearSVR(), param_grid={'C': 10.0 ** np.arange(-4, 2)}, **grid_search_kwargs),
    'RBF SVM': GridSearchCV(svm.SVR(), param_grid={'C': 10.0 ** np.arange(-2, 5), 'gamma': 10.0 ** np.arange(-5, 0)}, **grid_search_kwargs),
    'Elastic Net': GridSearchCV(ElasticNet(), param_grid={'alpha': 10.0 ** np.arange(-5, 1), 'l1_ratio': np.arange(0.1, 1.0, 0.2)}, **grid_search_kwargs),
    'Random Forrest': GridSearchCV(RandomForestRegressor(), param_grid={'n_estimators': [10, 25, 50, 100]}, **grid_search_kwargs),
    'Ada Boost': GridSearchCV(AdaBoostRegressor(), param_grid={'n_estimators': [10, 25, 50, 100], 'learning_rate': 10.0 ** np.arange(-2, 2)}, **grid_search_kwargs),
    'Ridge': GridSearchCV(Ridge(), param_grid={'alpha': 10.0 ** np.arange(0, 6)}, **grid_search_kwargs),
    'Lasso': GridSearchCV(Lasso(), param_grid={'alpha': 10.0 ** np.arange(-5, 1)}, **grid_search_kwargs),
    'Extra Random Forrest': GridSearchCV(ExtraTreesRegressor(), param_grid={'n_estimators': [10, 25, 50, 100]}, **grid_search_kwargs),
    'Poly SVM': GridSearchCV(svm.SVR(kernel='poly'), param_grid={'C': 10.0 ** np.arange(-1, 6), 'gamma': 10.0 ** np.arange(-3, 2)}, **grid_search_kwargs),
    'Sigmoid SVM': GridSearchCV(svm.SVR(kernel='sigmoid'), param_grid={'C': 10.0 ** np.arange(-3, 4), 'gamma': 10.0 ** np.arange(-6, -1)}, **grid_search_kwargs),
    'Neural Net': GridSearchCV(MLPRegressor(solver='lbfgs'), param_grid={'alpha': 10.0 ** -np.arange(0, 10), 'hidden_layer_sizes': [(50, 25, 15, 10), (50, 20, 15, 10, 5), (40, 25, 18, 12, 6), (38, 22, 16, 11, 8, 5)]}, **grid_search_kwargs),
}

plot_score_heatmaps(clfs)
