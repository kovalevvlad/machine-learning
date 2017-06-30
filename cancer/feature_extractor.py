import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
from data import train_df


feature_extractor = DataFrameMapper([
    ('Text', TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000, strip_accents="ascii")),
    ('Gene', CountVectorizer()),
    # TODO: sort out Variation field, see ideas.md for more info
    # (['Variation'], [CategoricalImputer(), MultiLabelBinarizer()])
])