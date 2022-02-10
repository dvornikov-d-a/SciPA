import config as c
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance
from data_preparing.dict_list_worker import to_list_of_dicts_of_series
from naive_bayes_mod import NaiveBayesMod
from metrics import calc_metrics, calc_ms
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance


import math
import time
import pandas as pd
import json


# Словарь: объём -> список всех результатов (метрик)
vol_metric_list = {}

dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))


for shuffle_number in range(c.shuffle_count):
    dfs_shuffled = shuffle(dfs, random_state=shuffle_number)
    dfs_balanced_shuffled = do_balance(shuffle(dfs, random_state=shuffle_number), c.field_class_, c.classes)
    # Размер обучающей выборки 4, 6, 8, ..., 100
    # Имитация итерационного процесса работы пользователя с системой
    for train_df_half_size in range(2, c.threshold_train_size + 2, 2):
        print()
        print('-----------------------------------------------')
        print(f'Перетасовка №{shuffle_number + 1}')
        print(f'Объём половинной выборки: {train_df_half_size}')
        if train_df_half_size not in vol_metric_list.keys():
            vol_metric_list[train_df_half_size] = []
        train_dfs, test_dfs = dict(zip([type_ for type_ in dfs_balanced_shuffled.keys()],
                                       [df.loc[:train_df_half_size - 1]
                                       .append(df.loc[len(df) - train_df_half_size:])
                                        for df in dfs_balanced_shuffled.values()])),\
                              dict(zip([type_ for type_ in dfs_balanced_shuffled.keys()],
                                       [df.loc[train_df_half_size:len(df) - train_df_half_size - 1]
                                        for df in dfs_balanced_shuffled.values()]))
        naive_bayes = NaiveBayesMod(c.best_alpha, c.best_w)
        train_Xs, train_ys = dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                   for type_, df in train_dfs.items()]), \
                             list(train_dfs.values())[0][c.field_class_].to_list()
        naive_bayes.fit(train_Xs, train_ys)
        naive_ys, true_ys = [naive_bayes.classify(X)
                             for X in
                             to_list_of_dicts_of_series(dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                                              for type_, df in test_dfs.items()]))], \
                            list(test_dfs.values())[0][c.field_class_].to_list()
        metrics = calc_metrics(naive_ys, true_ys)
        vol_metric_list[train_df_half_size].append(metrics)
        for metric_name, value in metrics.items():
            print(f'{metric_name}:\t{value}')

        naive_estimations = [naive_bayes.estimate(X)
                             for id_, X in
                             zip(to_list_of_dicts_of_series(dict([(type_, df[c.field_id_])
                                                              for type_, df in test_dfs.items()])),
                                 to_list_of_dicts_of_series(dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                                              for type_, df in test_dfs.items()])))]
        scores = [naive_estimation[c.classes[1]] - naive_estimation[c.classes[0]]
                  for naive_estimation in naive_estimations]
        max_score = max(scores)
        for i in range(train_df_half_size):
            scores.insert(0, max_score + 1)
            scores.append(max_score + 1)
        scores_ = 'scores_'
        for s in c.set_names:
            dfs_balanced_shuffled[s][scores_] = scores
            dfs_balanced_shuffled[s].sort_values(scores_, ascending=False)
            dfs_balanced_shuffled[s].drop(scores_, axis=1, inplace=True)

vol_ms = {}
for train_vol, metric_list in vol_metric_list.items():
    vol_ms[train_vol] = calc_ms(metric_list)

with open(c.five_dive_json, 'w', encoding=c.encoding) as f:
    f.write(json.dumps(vol_ms))

