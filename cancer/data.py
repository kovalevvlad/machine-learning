import pandas as pd
import pkg_resources
import os


def raw_data_file_path(file_name):
    file_path = pkg_resources.resource_filename(__name__, os.path.join("raw_data", file_name))
    return file_path


def read_csv(file_name, sep=",", custom_header=None):
    cache_file_name = file_name + ".feather"
    cache_file_path = raw_data_file_path(cache_file_name)
    if os.path.exists(cache_file_path):
        return pd.read_feather(cache_file_path)
    else:
        file_path = raw_data_file_path(file_name)
        assert os.path.exists(file_path), "{} does not exist. Are you sure you have downloaded this file from Kaggle?".format(file_path)
        # engine='python' is required to suppress the 'Falling back to the 'python' engine because the 'c' engine
        # does not support regex separators' warning caused by '||' delimiter
        extra_args = dict() if custom_header is None else {'skiprows': 1, 'header': None, 'names': custom_header}
        file_data = pd.read_csv(file_path, sep=sep, engine='python', **extra_args)
        # Why do we use a cache? Reading the entire dataset without a cache takes 56 seconds,
        # but only 22 milliseconds with a cache! I know!
        file_data.to_feather(cache_file_path)
        return file_data


train_variants = read_csv("training_variants.zip")
train_text = read_csv("training_text.zip", sep=r"\|\|", custom_header=['ID', 'Text'])
merged_train_data = pd.merge(train_variants, train_text, on="ID", how='outer')
train_df = merged_train_data[['Text', 'Gene', 'Variation']].applymap(lambda x: x.decode('utf-8').encode('ascii', 'ignore'))
train_y = merged_train_data['Class'].values

test_variants = read_csv("test_variants.zip")
test_text = read_csv("test_text.zip", sep="\|\|", custom_header=['ID', 'Text'])
merged_test_data = pd.merge(test_variants, test_text, on="ID", how='outer')
test_df = merged_test_data[['Text', 'Gene', 'Variation']].applymap(lambda x: x.decode('utf-8').encode('ascii', 'ignore'))
test_id = merged_test_data['ID']
