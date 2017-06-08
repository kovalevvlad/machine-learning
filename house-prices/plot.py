import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def _round_with_sig_figs(n, x):
    round_to_digits = list(-np.floor(np.log10(x)) + (n - 1))
    return np.array([round(float(x_), int(round_to_digits_)) for x_, round_to_digits_ in zip(x, round_to_digits)])


def _round_floats(possibly_float_array):
    if np.issubdtype(possibly_float_array.dtype, np.number):
        return _round_with_sig_figs(3, possibly_float_array)
    else:
        return possibly_float_array


def features_with_weight(feature_weight_series, title):
    assert isinstance(feature_weight_series, pd.Series)
    feature_plot = feature_weight_series.plot(kind='barh', legend=False)
    feature_plot.set_title(title)
    feature_plot.set_xlabel('Feature Mutual Information with the House Price')
    feature_plot.set_ylabel('Feature')
    plt.tight_layout()
    plt.show()


def score_heatmaps(grid_searches, title, X_data, y_data, rotate_x_axis_by=None, features_excluded_from_plot=set(), clipping_score=None):
    subplot_count = len(grid_searches)
    n_columns = int(np.ceil(np.sqrt(float(subplot_count))))
    n_rows = subplot_count / n_columns + (0 if subplot_count % n_columns == 0 else 1)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False)

    for ax, (estimator_name, grid_search) in zip(axes.flat, grid_searches.items()):
        print "Fitting {}".format(estimator_name)
        grid_search.fit(X_data, y_data)
        df = pd.DataFrame([dict(param_dict.items() + [('score', score)]) for score, param_dict in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params'])])
        df['score'] = df.score.clip(lower=clipping_score)
        params = list(set(df.columns) - {'score'})
        # Only care about params with more than one value
        params = [p for p in params if len(df[p].unique()) > 1 and p not in features_excluded_from_plot]
        assert len(params) in (1, 2)
        if len(params) == 1:
            param_name = params[0]
            im_data = np.array([df.score]).T
            cax = ax.imshow(im_data, vmin=df.score.min(), vmax=df.score.max(), aspect='auto', extent=(0, 1, 0, len(df.score)), origin='lower', cmap='plasma')
            best_score_ix = im_data.argmax()
            ax.text(0.5, best_score_ix + 0.5, '{0:.3f}'.format(float(im_data[best_score_ix])))
            ax.set_ylabel(param_name)
            # for some reason matplotlib refuses to display the first tick... shift it along!
            ticks = _round_floats(df[param_name].values)
            ax.set_yticklabels(ticks)
            ax.set_yticks(0.5 + np.arange(len(ticks)))
            ax.axes.get_xaxis().set_visible(False)
        if len(params) == 2:
            ix_param = params[0]
            col_param = params[1]
            pivoted_df = df[[c for c in df.columns if c not in features_excluded_from_plot]].pivot(index=ix_param, columns=col_param)
            im_data = pivoted_df.values
            cax = ax.imshow(im_data, vmin=df.score.min(), vmax=df.score.max(), aspect='auto', extent=(0, len(pivoted_df.columns.levels[1]), 0, len(pivoted_df.index)), origin='lower', cmap='plasma')
            best_score_x, best_score_y = np.unravel_index(im_data.argmax(), im_data.shape)
            ax.text(best_score_y + 0.1, best_score_x + 0.5, '{0:.3f}'.format(float(im_data[best_score_x, best_score_y])))
            ax.set_ylabel(ix_param)
            ax.set_xlabel(col_param)
            yticks = _round_floats(pivoted_df.index.values)
            ax.set_yticklabels(yticks)
            ax.set_yticks(0.5 + np.arange(len(yticks)))
            xticks = _round_floats(pivoted_df.columns.levels[1])
            has_long_labels = xticks.dtype.name == 'object' and max([len(str(tick)) for tick in xticks]) > 5
            rotation = rotate_x_axis_by if rotate_x_axis_by is not None else (15. if has_long_labels else 'horizontal')
            alignment = 'right' if has_long_labels else 'center'
            ax.set_xticklabels(xticks, rotation=rotation, horizontalalignment=alignment)
            ax.set_xticks(0.5 + np.arange(len(xticks)))

        ax.set_title(estimator_name)
        ax.grid(False)
        colorbar = plt.colorbar(cax, ax=ax)
        colorbar.set_label("Cross-Validation Score")

    hidden_axes = axes.flat[len(grid_searches):]
    for hidden_axis in hidden_axes:
        hidden_axis.set_visible(False)

    plt.suptitle(title)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.show()
