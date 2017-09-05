import os
import pandas as pd
import pkg_resources
import numpy as np


def raw_data_file_path(file_name):
    file_path = pkg_resources.resource_filename(__name__, os.path.join("raw_data", file_name))
    return file_path


def generate_with_caching(file_name, generator_func):
    cache_file_name = file_name + ".feather"
    cache_file_path = raw_data_file_path(cache_file_name)
    if os.path.exists(cache_file_path):
        return pd.read_feather(cache_file_path)
    else:
        file_data = generator_func()
        file_data.to_feather(cache_file_path)
        return file_data


def read_csv(file_name):
    def loader_func():
        file_path = raw_data_file_path(file_name)
        assert os.path.exists(file_path), "{} does not exist. Are you sure you have downloaded this file from Kaggle?".format(file_path)
        return pd.read_csv(file_path, encoding="utf-8")

    return generate_with_caching(file_name, loader_func)


def read_key_df():
    key_df = read_csv("key_1.csv.zip")

    def generator_func():
        return pd.DataFrame({u"Id": key_df["Id"].values, u"date": pd.to_datetime(key_df.Page.str[-10:]).values, u"page": key_df.Page.str[:-11].values})

    return generate_with_caching("key_df_processed", generator_func)


def read_train_df():
    train_df = read_csv("train_1.csv.zip")

    def generator_func():
        tdf = train_df.set_index("Page").T
        tdf.index = pd.to_datetime(tdf.index)
        tdf = tdf.reset_index()
        tdf = tdf.rename(columns={"index": u"index"})
        return tdf

    return generate_with_caching("train_df_processed", generator_func).set_index(u"index")


train_df = read_train_df().fillna(0).astype(np.int32)
