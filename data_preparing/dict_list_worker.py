def to_list_of_dicts_of_series(dict_of_dataframes):
    list_ = []
    objects_count = len(list(dict_of_dataframes.values())[0])
    for i in range(objects_count):
        dict_ = {}
        for key, df in dict_of_dataframes.items():
            series_ = df.iloc[i]
            dict_[key] = series_
        list_.append(dict_)
    return list_


