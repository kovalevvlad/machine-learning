import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper

from data import raw_data_file_path, train_df


# Obtained from http://www.genenames.org/cgi-bin/statistics
# interesting_columns = ['symbol', 'gene_family_id', 'location']
gene_info_df = pd.read_csv(raw_data_file_path("gene_info.csv"), sep="\t")


def location_features(gene_df):
    """
    Gene location string specification:
    https://www.ncbi.nlm.nih.gov/Class/MLACourse/Original8Hour/Genetics/chrombanding.html
    """
    location = gene_df.set_index("symbol").location
    spot_location_regex = r"^(?P<chromosome>(?:[0-9]{1,2}|[XY]))(?P<hand>q|p)(?P<band>[0-9])(?P<subband>[0-9])(?:\.(?P<subsubband>[0-9]{1,3}))?$"
    range_location_regex = r"^(?P<chromosome>(?:[0-9]{1,2}|[XY]))(?P<hand>q|p)(?P<band>[0-9])(?P<subband>[0-9])(?:\.(?P<subsubband>[0-9]{1,3}))?-(?P<second_hand>q|p)(?P<second_band>[0-9])(?P<second_subband>[0-9])(?:\.(?P<second_subsubband>[0-9]{1,3}))?$"
    most_common_good_location = location[location.str.match(spot_location_regex)].value_counts().index[0]

    # Removing crappy locations such as - "22 not on reference assembly"
    location[(~location.str.match(spot_location_regex)) & (~location.str.match(range_location_regex))] = most_common_good_location

    # deal with spot locations
    spot_locations = location[location.str.match(spot_location_regex)]
    spot_location_data = pd.concat([spot_locations.to_frame(), spot_locations.str.extract(spot_location_regex, expand=True)], axis=1)

    # deal with range locations
    range_locations = pd.DataFrame({"location": location[location.str.match(range_location_regex)]})
    range_location_data = pd.concat([range_locations, range_locations.location.str.extract(range_location_regex, expand=True)], axis=1)
    assert spot_location_data.shape[0] + range_location_data.shape[0] == location.shape[0]

    # For now do the simplest thing for range locations, pick the beginning of the range as the spot location
    # In the future might take the middle?
    columns_to_keep = [c for c in range_location_data.columns if not c.startswith("second_")]
    all_location_data = pd.concat([spot_location_data, range_location_data[columns_to_keep]])

    # For now ignore subsubband
    del all_location_data["subsubband"]
    # Drop location itself since it's not a feature
    del all_location_data["location"]

    # encode all features using numeric values
    all_location_data['hand'] = (all_location_data['hand'] == "q")
    all_location_data['chromosome'] = all_location_data['chromosome'].apply(lambda chromosome: int(chromosome) if chromosome.isdigit() else (23 if chromosome == "X" else 24))

    transform = DataFrameMapper([
        (["chromosome"], OneHotEncoder()),
        (["band"], OneHotEncoder()),
        (["subband"], OneHotEncoder()),
        (["hand"], None)])

    transformed_features = transform.fit_transform(all_location_data)
    return pd.DataFrame(transformed_features, columns=transform.transformed_names_, index=all_location_data.index)


def gene_family_features(gene_info, genes_of_interest):
    # It makes sense to throw away all gene groups not in the training data right away
    # This reduces the number of family group features by a factor of 6
    interesting_gene_subset = pd.merge(gene_info, genes_of_interest, left_on="symbol", right_on="Gene", how="inner")
    vectorizer = CountVectorizer(binary=True).fit(interesting_gene_subset.gene_family_id.fillna(""))
    family_group_features = vectorizer.transform(gene_info.gene_family_id.fillna(""))
    columns = ["gene_family_" + family_id for family_id, _ in sorted(vectorizer.vocabulary_.items(), key=(lambda x: x[1]))]
    return pd.DataFrame(family_group_features.todense(), columns=columns, index=gene_info["symbol"])


features = pd.concat([location_features(gene_info_df), gene_family_features(gene_info_df, train_df[['Gene']])], axis=1)
