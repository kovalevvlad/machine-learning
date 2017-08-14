def safe_reindex(df, new_index):
    assert set(df.index) == set(new_index)
    return df.reindex(index=new_index)
