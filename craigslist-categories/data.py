import pandas as pd
from StringIO import StringIO
from scipy.sparse import hstack
import numpy as np
import inflect

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

inflect_engine = inflect.engine()


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
    df = read_stream(f)

    category_encoder = LabelEncoder().fit(df.category)
    y = category_encoder.transform(df.category)

    city_encoder = LabelEncoder().fit(df.city)
    section_encoder = LabelEncoder().fit(df.section)

    no_punctuation = df.heading.str.lower().apply(drop_punctuation)

    count_transformation = CountVectorizer(ngram_range=(1, 2), binary=True).fit(no_punctuation)
    X = hstack([np.array([city_encoder.transform(df.city),
                          section_encoder.transform(df.section)]).T,
                          count_transformation.transform(no_punctuation)])
