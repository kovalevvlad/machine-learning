from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
import pandas as pd
import gene_info


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


feature_extractor = DataFrameMapper([
    ('Text', Pipeline([
        ("vectorize", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, strip_accents="ascii")),
        ("dim_red", TruncatedSVD(n_components=50, n_iter=25))])),
    # Gene features don't seem to do anything now. I have a strong suspicion that they will become important once we
    # start using the variation information.
    ('Gene', GeneFeatureExtractor()),
    # TODO: sort out Variation field, see ideas.md for more info
    # (['Variation'], [CaetegoricalImputer(), MultiLabelBinarizer()])
])