from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn_pandas import DataFrameMapper
import pandas as pd
import gene_info
import numpy as np
from categorical_one_hot import CategoricalOneHotEncoder


class GeneFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = gene_info.features.columns

    def transform(self, X, y=None, copy=None):
        key = "symbol"

        # Unknown gene imputation
        input_gene_series = pd.Series(X)
        unknown_genes = set(input_gene_series.values) - set(gene_info.features.index.values)
        most_common_gene = input_gene_series.value_counts().index[0]
        input_gene_series.loc[input_gene_series.apply(lambda gene: gene in unknown_genes)] = most_common_gene

        # Order-preserving join
        input_gene_df = pd.DataFrame({key: input_gene_series.values, "order": range(X.shape[0])})
        unordered_features = pd.merge(gene_info.features, input_gene_df, left_index=True, right_on=key)
        ordered_features = unordered_features.sort_values("order")
        # Assert order has not changed
        assert (pd.Series(input_gene_df[key].values) == pd.Series(ordered_features[key].values)).all()
        return ordered_features[gene_info.features.columns].values

    def fit(self, X, y=None):
        return self


class VariationFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.change_variation_transform = DataFrameMapper([
            ("from_protein", CategoricalOneHotEncoder(allow_nulls=True)),
            ("to_protein", CategoricalOneHotEncoder(allow_nulls=True)),
            (["variation_location"], None)])
        self.variation_type_transform = DataFrameMapper([(["variation_type"], MultiLabelBinarizer())])
        self.classes_ = None

    @staticmethod
    def variation_type_vector(is_change_variation_vector, variation):
        known_variation_types = {
            "truncating mutations",
            "deletion",
            "amplification",
            "fusions",
            "fusion",
            "overexpression",
            "change"
        }
        variation_type = variation.copy()
        variation_type.loc[is_change_variation_vector] = "change"
        # TODO: Keep fusion information in the features
        variation_type.loc[variation.str.endswith(" fusion")] = "fusion"
        variation_type.loc[~variation_type.isin(known_variation_types)] = "other"
        return pd.DataFrame({"variation_type": variation_type.values})

    @staticmethod
    def change_variation_features(variation):
        # TODO: What does a * protein mean?
        change_variation_regex = "(?P<from_protein>[a-z*])(?P<variation_location>[0-9]+)(?P<to_protein>[a-z*])"
        change_variation_features = variation.str.extract(change_variation_regex, expand=True)

        # filling non-change variations with -1 in this field
        change_variation_features["variation_location"] = change_variation_features["variation_location"].fillna(-1)

        is_change_variation_vector = variation.str.match(change_variation_regex)
        return change_variation_features, is_change_variation_vector

    def transform(self, X, y=None, copy=None):
        variation = pd.Series(X)
        variation = variation.str.lower()

        change_variation_features, is_change_variation_vector = self.change_variation_features(variation)
        change_variation_features_transformed = self.change_variation_transform.transform(change_variation_features)

        variation_type = self.variation_type_vector(is_change_variation_vector, variation)
        variation_type_features = self.variation_type_transform.transform(variation_type)

        self.classes_ = self.change_variation_transform.transformed_names_ + self.variation_type_transform.transformed_names_

        return np.hstack([change_variation_features_transformed, variation_type_features])

    def fit(self, X, y=None):
        variation = pd.Series(X)
        variation = variation.str.lower()

        change_variation_features, is_change_variation_vector = self.change_variation_features(variation)
        # Only fit on change variations
        self.change_variation_transform.fit(change_variation_features)

        variation_type = self.variation_type_vector(is_change_variation_vector, variation)
        self.variation_type_transform.fit(variation_type)

        return self


feature_extractor = DataFrameMapper([
    ('Text', Pipeline([
        ("vectorize", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, strip_accents="ascii")),
        ("dim_red", TruncatedSVD(n_components=50, n_iter=25))])),
    # TODO: These features don't make a big difference for the prediction performance. Why?!
    ('Gene', GeneFeatureExtractor()),
    ('Variation', Pipeline([("extract", VariationFeatureExtractor()), ("dim_red", TruncatedSVD(n_components=25, n_iter=10))]))
])
