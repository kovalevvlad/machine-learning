import pandas as pd
from StringIO import StringIO
from scipy.sparse import hstack
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, scale
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from sklearn.preprocessing import MinMaxScaler


def key_for_value(dictionary, value):
    return {v: k for k, v in dictionary.items()}[value]


def unzip(iterables):
    return zip(*iterables)


def read_stream(s):
    lines_to_load = int(s.readline())
    string_io = StringIO()
    string_io.write('[')
    string_io.writelines([s.readline() + ("" if i == lines_to_load - 1 else ",") for i in xrange(lines_to_load)])
    string_io.write(']')
    string_io.seek(0)
    return pd.read_json(string_io)


def drop_punctuation(heading):
    # drop all digits and punctuation - iphone4 is not a great feature but phone is.
    letters_only = ''.join([c if (97 <= ord(c) <= 122 or c == ' ') else ' ' for c in heading])
    words = [x for x in letters_only.split(" ") if len(x) > 0]
    return ' '.join(words)


def stem_words(heading, stemmer):
    words = heading.split(' ')
    return ' '.join(stemmer(w) for w in words)


with open("data.json") as f:
    # This weird approach is taken to allow submitting this solution to hackerrank where
    # input is provided via stdin
    df = read_stream(f)

    y = LabelEncoder().fit_transform(df.category)

    city_feature_transform = CountVectorizer(token_pattern='.+')
    section_feature_transform = CountVectorizer(token_pattern='.+')

    title_length = df.heading.str.len().astype(np.float32)
    number_of_punctuation_chars = df.heading.str.count("[^a-zA-Z ]").astype(np.float32)
    ratio_of_punctuation_chars = number_of_punctuation_chars / title_length
    number_of_capital_chars = df.heading.str.count("[A-Z]").astype(np.float32)
    ratio_of_capital_chars = number_of_capital_chars / title_length

    no_punctuation = df.heading.str.lower().apply(drop_punctuation)
    nltk.download("wordnet")
    wordnet = WordNetLemmatizer()
    stemmed = no_punctuation.apply(lambda x: stem_words(x, wordnet.lemmatize))

    title_feature_transform = CountVectorizer(ngram_range=(1, 3), binary=True)
    city_features = city_feature_transform.fit_transform(df.city)
    section_features = section_feature_transform.fit_transform(df.section)
    title_word_features = title_feature_transform.fit_transform(stemmed)
    adhoc_features = MinMaxScaler().fit_transform(np.array([title_length,
                                                            number_of_capital_chars,
                                                            number_of_punctuation_chars,
                                                            ratio_of_capital_chars,
                                                            ratio_of_punctuation_chars]).T)

    X = hstack([adhoc_features,
                city_features,
                section_features,
                title_word_features])

    adhoc_feature_names = ['title_length',
                           'number_of_capital_chars',
                           'number_of_punctuation_chars',
                           'ratio_of_capital_chars',
                           'ratio_of_punctuation_chars']

    feature_labelers = [
        (adhoc_features, lambda ix: adhoc_feature_names[ix]),
        (city_features, lambda ix: 'city: ' + key_for_value(city_feature_transform.vocabulary_, ix)),
        (section_features, lambda ix: 'section: ' + key_for_value(section_feature_transform.vocabulary_, ix)),
        (title_word_features, lambda ix: 'word: ' + key_for_value(title_feature_transform.vocabulary_, ix))]

    def feature_ix_to_name(ix):
        labeler_width, labelers = unzip([(features.shape[1], labeler) for features, labeler in feature_labelers])
        labeler_ceiling_index = np.cumsum(labeler_width).tolist()
        labeler_floor_index = [0] + labeler_ceiling_index
        labelers_with_indices = zip(labelers, labeler_floor_index, labeler_ceiling_index)
        labeler, floor = [(l, f) for l, f, c in labelers_with_indices if ix < c][0]
        offset = ix - floor
        return labeler(offset)
