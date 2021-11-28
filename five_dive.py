import config as c
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance
from data_preparing.dict_list_worker import to_list_of_dicts_of_series
from naive_bayes_mod import NaiveBayesMod
from metrics import calc_metrics

import math
import time
import pandas as pd

dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))
# dfs_shuffled_balanced = do_balance(shuffle(dfs, random_state=1), c.field_class_, c.classes)
ys = [y for y in list(dfs.values())[0][c.field_class_]]
print(f'Доля правильных ответов в выдаче: {len([y for y in ys if y == 1]) / len(ys)}')
print()

border_indexes = (10, 9, 8, 7, 6, 5, 4, 3, 2)
for border_index in border_indexes:
    print(f'Объъём обучающей выборки: {border_index * 2}')
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
    for metric_name, value in metrics.items():
        print(f'{metric_name}:\t{value}')
    print('--------------------------')
