from sklearn.feature_extraction.text import CountVectorizer

from data_collection.sql_data_source import SqlLiteDataSource
import pandas as pd
from scipy import stats
from sklearn.metrics import mutual_info_score
from matplotlib import pyplot as plt


with SqlLiteDataSource("data_collection/data.sqlt") as sql_source:
    df = pd.read_sql("select title, price, rank from listing", sql_source.connection)
    rank = df["rank"]
    vectorizer = CountVectorizer(stop_words="english", max_features=50000, binary=True, ngram_range=(1, 2))
    title_features = vectorizer.fit_transform(df.title)
    columns = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
    title_features_df = pd.DataFrame(title_features.todense(), columns=columns)
    s = title_features_df.sum()
    # Ensure we have enough data
    frequent_phrases = s[s > 9]
    features = title_features_df[frequent_phrases.index.values]
    features["__len__"] = df.title.str.len()
    features["__word_count__"] = df.title.str.count(" ")
    features["__capitals__"] = df.title.str.count("[A-Z]")
    mutual_info = features.apply(lambda feature: mutual_info_score(rank, feature))
    r_val = features.apply(lambda feature: stats.linregress(feature, rank)[2])
    features_by_influence = pd.DataFrame({"r": r_val, "mi": mutual_info}).sort_values("mi")
    # features with the strongest negative effect on the rank (what you want)
    best_features = features_by_influence[features_by_influence.r < 0].tail(25)
    # features with the strongest positive effect on the rank (what you do not want)
    worst_features = features_by_influence[features_by_influence.r > 0].tail(25)

    print "Best Features:\n {}".format(best_features.to_string())
    print "Worst Features:\n {}".format(worst_features.to_string())

    binned_prices = pd.DataFrame({"rank": df["rank"], "interval": pd.cut(df.price, pd.np.arange(5.75, 30.75, 0.5)).apply(lambda i: i.left)})
    interval_sizes = binned_prices.groupby("interval").size()
    significant_intervals = set(interval_sizes[interval_sizes > 9].index.values)
    binned_prices_for_significant_intervals = binned_prices[binned_prices["interval"].isin(significant_intervals)]

    # Plotting rank distribution per price bucket. This should help picking the optimal price for the shirts.
    binned_prices[binned_prices["interval"].isin(significant_intervals)].boxplot(by="interval")
    plt.show()