from frozendict import frozendict

import config as c
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance
from data_preparing.dict_list_worker import to_list_of_dicts_of_series
from naive_bayes_mod import NaiveBayesMod
from metrics import calc_metrics, calc_ms


import math
import time
import pandas as pd
import json


dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))
# dfs_shuffled_balanced = do_balance(shuffle(dfs, random_state=1), c.field_class_, c.classes)
ys = [y for y in list(dfs.values())[0][c.field_class_]]
print(f'Доля правильных ответов в выдаче: {len([y for y in ys if y == 1]) / len(ys)}')
print()

# Словарь: объём -> список всех результатов (метрик)
vol_metric_list = {}

border_indexes = set(range(1, 196))
# Для объёма обучающей выборки (от 2 до 390)
for border_index in border_indexes:
    train_df_size = 2 * border_index
    print(f'Объём обучающей выборки: {train_df_size}')
    if train_df_size not in vol_metric_list.keys():
        vol_metric_list[train_df_size] = []
    # Обучающая выборка: равное количество документов из начала и конца выдачи
    # Тестовая выборка: все остальные документы (в середине выдачи)
    train_dfs, test_dfs = dict(zip([type_ for type_ in dfs.keys()],
                                   [df.loc[:border_index]
                                   # .append(df.loc[math.floor(len(df)/2)-border_index+1:math.floor(len(df)/2)])
                                   .append(df.loc[len(df)-border_index+1:]) for df in dfs.values()])), \
                          dict(zip([type_ for type_ in dfs.keys()],
                                   [df.loc[border_index+1:len(df)-border_index]
                                   # .append(df.loc[math.floor(len(df)/2)+1:len(df)-border_index])
                                    for df in dfs.values()]))
    naive_bayes = NaiveBayesMod(c.best_alpha, c.best_w)
    train_Xs, train_ys = dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                               for type_, df in train_dfs.items()]), \
                         list(train_dfs.values())[0][c.field_class_].to_list()
    naive_bayes.fit(train_Xs, train_ys)
    naive_ys, true_ys = [naive_bayes.classify(X)
                         for X in to_list_of_dicts_of_series(dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                                                   for type_, df in test_dfs.items()]))], \
                        list(test_dfs.values())[0][c.field_class_].to_list()
    metrics = calc_metrics(naive_ys, true_ys)
    vol_metric_list[train_df_size].append(metrics)
    for metric_name, value in metrics.items():
        print(f'{metric_name}:\t{value}')
    print('--------------------------')

# Результаты по отложенной выборке
vol_ms = {}
for train_vol, metric_list in vol_metric_list.items():
    vol_ms[train_vol] = calc_ms(metric_list)

with open('res_null_coef/min_fit.json', 'w', encoding=c.encoding) as f:
    f.write(json.dumps(vol_ms))

