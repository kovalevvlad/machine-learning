import pandas as pd
import pkg_resources
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np


def transform_raw_feature_df(df_raw):
    df = df_raw[[c for c in df_raw.columns if c != 'Id']]
    dtypes = df.dtypes

    # Order is important when it comes to inverse_transform
    magnitude_feature_names = sorted(dtypes[dtypes != 'object'].index.values)
    category_feature_names = sorted(set(df.columns.values) - set(magnitude_feature_names))

    # TODO: Deal with hidden NANs (e.g. PoolArea = 0)
    magnitude_features = df[magnitude_feature_names].fillna(0)
    scaled_magnitude_features = StandardScaler().fit_transform(magnitude_features.values)

    # Pandas parses 'NA' as np.nan, recovering the desired state here
    category_features = df[category_feature_names].fillna('NA')
    one_hot_category_feature_transforms = [CountVectorizer(token_pattern='.+').fit(feature)
                                           for _, feature in category_features.iteritems()]
    one_hot_category_features = [transform.transform(raw_feature).todense()
                                 for transform, (feature_name, raw_feature)
                                 in zip(one_hot_category_feature_transforms, category_features.iteritems())]

    def category_names_in_order(one_hot_transform):
        map_sorted_by_ix = sorted(one_hot_transform.vocabulary_.items(), key=lambda x: x[1])
        return zip(*map_sorted_by_ix)[0]

    all_features = np.array(np.hstack([scaled_magnitude_features] + one_hot_category_features))
    category_feature_names = ["{}: {}".format(f_name, f_value) for f_name, one_hot_transform in
                              zip(category_feature_names, one_hot_category_feature_transforms) for f_value in
                              category_names_in_order(one_hot_transform)]
    feature_names = magnitude_feature_names + category_feature_names
    return all_features, feature_names

test_file = pkg_resources.resource_filename(__name__, 'test.csv')
train_file = pkg_resources.resource_filename(__name__, 'train.csv')

df_test = pd.read_csv(test_file)
df_train = pd.read_csv(train_file)

y_train = df_train.SalePrice.values
del df_train['SalePrice']
X_train, train_feature_names = transform_raw_feature_df(df_train)
X_test, _ = transform_raw_feature_df(df_test)
