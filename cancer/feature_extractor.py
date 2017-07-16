from sklearn import pipeline, feature_extraction, decomposition
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
import pandas as pd
import gene_info
from categorical_one_hot import CategoricalOneHotEncoder
import numpy as np


class GeneFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.classes_ = gene_info.features.columns

    def transform(self, X, y=None, copy=None):
        # Unknown gene imputation
        input_gene_series = pd.Series(X)
        unknown_genes = set(input_gene_series.values) - set(gene_info.features.index.values)
        most_common_gene = input_gene_series.value_counts().index[0]
        input_gene_series.loc[input_gene_series.isin(unknown_genes)] = most_common_gene

        # Order-preserving join
        key = "symbol"
        input_gene_df = pd.DataFrame({key: input_gene_series.values, "order": range(X.shape[0])})
        unordered_features = pd.merge(gene_info.features, input_gene_df, left_index=True, right_on=key)
        ordered_features = unordered_features.sort_values("order")

        # Assert order has not changed
        assert (pd.Series(input_gene_series.values) == pd.Series(ordered_features[key].values)).all()
        return ordered_features[gene_info.features.columns].values

    def fit(self, X, y=None):
        return self


class LabelEncoderIgnoringY(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def transform(self, X, y=None, copy=None):
        return self.encoder.transform(X)

    def fit(self, X, y=None):
        self.encoder.fit(X)
        return self


class VariationFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_transform = DataFrameMapper([
            ("from_protein", CategoricalOneHotEncoder(allow_nulls=True)),
            ("to_protein", CategoricalOneHotEncoder(allow_nulls=True)),
            ("location", None),
            ("type", CategoricalOneHotEncoder(allow_nulls=False))])
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
        return pd.DataFrame({"type": variation_type.values})

    @staticmethod
    def change_variation_features(variation):
        # TODO: What does a * protein mean?
        change_variation_regex = "(?P<from_protein>[a-z*])(?P<location>[0-9]+)(?P<to_protein>[a-z*])"
        change_variation_features = variation.str.extract(change_variation_regex, expand=True)

        # filling non-change variations with -1 in this field
        change_variation_features["location"] = change_variation_features["location"].fillna(-1).astype(np.int32)

        is_change_variation_vector = variation.str.match(change_variation_regex)
        return change_variation_features, is_change_variation_vector

    def transform(self, X, y=None, copy=None):
        variation = pd.Series(X)
        variation = variation.str.lower()

        change_variation_features, is_change_variation_vector = self.change_variation_features(variation)
        variation_type = self.variation_type_vector(is_change_variation_vector, variation)
        variation_features = pd.concat([variation_type, change_variation_features], axis=1)
        variation_features_transformed = self.feature_transform.transform(variation_features)

        self.classes_ = self.feature_transform.transformed_names_

        return variation_features_transformed

    def fit(self, X, y=None):
        variation = pd.Series(X)
        variation = variation.str.lower()

        change_variation_features, is_change_variation_vector = self.change_variation_features(variation)
        variation_type = self.variation_type_vector(is_change_variation_vector, variation)

        all_features = pd.concat([change_variation_features, variation_type], axis=1)
        self.feature_transform.fit(all_features)

        return self


class StringLengthTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, y=None, copy=None):
        strings = pd.Series(X)
        return strings.str.len().values.reshape((-1, 1))

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return ["length"]


class StringRegexCountTransform(BaseEstimator, TransformerMixin):
    def __init__(self, regex):
        self.regex = regex

    def transform(self, X, y=None, copy=None):
        strings = pd.Series(X)
        return strings.str.count(self.regex).values.reshape((-1, 1))

    def fit(self, X, y=None):
        return self

    def get_feature_names(self):
        return ["count"]


class NamedPipeline(Pipeline):
    def get_feature_names(self):
        last_transformer = self.steps[-1][1]
        if "get_feature_names" in dir(last_transformer):
            return last_transformer.get_feature_names()
        elif "classes_" in dir(last_transformer):
            return last_transformer.classes_
        else:
            raise RuntimeError("Issue with {}".format(type(last_transformer)))


class NamedTruncatedSVD(TruncatedSVD):
    def get_feature_names(self):
        return [str(i) for i in range(self.n_components)]


feature_extractor = DataFrameMapper([
            ('Text', FeatureUnion([
                [("tf-idf", NamedPipeline([("words", TfidfVectorizer())])), ("dim-red", NamedTruncatedSVD(n_components=100, n_iter=15))],
                ("text-length", StringLengthTransform()),
                ("word-count", StringRegexCountTransform(" "))])),
            ('Gene', GeneFeatureExtractor()),
            ('Variation', VariationFeatureExtractor())
        ], df_out=True)
