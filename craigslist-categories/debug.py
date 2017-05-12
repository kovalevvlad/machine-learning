import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import learning_curve


# taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curves(estimators, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimators : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

        This can also be a dict or str -> object type that implements the "fit" and "predict" methods.
        keys of the dict will be used as estimator labels.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("{}Score".format("" if scoring is None else scoring + " "))
    plt.grid()

    estimators = estimators if isinstance(estimators, dict) else {'Estimator': estimators}
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for (estimator_name, current_estimator), color in zip(estimators.items(), cycle(colors)):
        train_sizes, train_scores, test_scores = learning_curve(current_estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color=color)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color=color)
        plt.plot(train_sizes, train_scores_mean, '--', color=color, label="{} Training Score".format(estimator_name))
        plt.plot(train_sizes, test_scores_mean, '-', color=color, label="{} CV Score".format(estimator_name))

    plt.legend(loc="best")
    return plt
