import math
import random
import pandas as pd
from frozendict import frozendict
import json

import config as c
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance
from data_preparing.k_folds_split import k_folds_split, folds_union
from data_preparing.dict_list_worker import to_list_of_dicts_of_series
from naive_bayes_mod import NaiveBayesMod
from metrics import calc_metrics, calc_ms


# Варианты альф
alphas = (1, 0.8, 0.5)

best_alpha = alphas[2]

# Варианты весов
ws = (
    # Слова, авторы, журналы, области
    frozendict(zip(c.set_names, (1, 1, 0, 0))),
    frozendict(zip(c.set_names, (1, 4, 3, 2)))
)

best_w = ws[0]

# Словарь: объём -> список всех результатов (метрик)
vol_metric_list = {}

dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))
# Несколько перемешиваний
for shuffle_number in range(c.shuffle_count):
    n = len(list(dfs.values())[0].index)  # 494
    k = math.floor(n / c.min_vol)  # 98

    dfs_shuffled_balanced = do_balance(shuffle(dfs, random_state=shuffle_number), c.field_class_, c.classes)
    # Разделение на максимальное количество блоков
    folds = k_folds_split(dfs_shuffled_balanced, k)

    # Диапазон количества блоков (объёмов) для обучающей выборки
    k_train_max = math.floor(c.max_vol_frac * k)  # 78
    k_train_min = 1

    # Фиксированное количество блоков (объём) для тестовой выборки
    k_test = k - k_train_max

    # Для текущего объёма обучающей выборки
    for k_train in range(k_train_max, k_train_min - 1, -1):
        train_vol = k_train * c.min_vol
        if train_vol not in vol_metric_list.keys():
            vol_metric_list[train_vol] = []
        # Несколько перемешиваний
        for sample_number in range(c.samples_count):
            print(f'Progress: {train_vol}: {shuffle_number}/{c.shuffle_count} | {sample_number}/{c.samples_count}')

            random.seed(sample_number)
            random.shuffle(folds)
            # Выделение обучающей и тестовой выборок
            train_dfs, test_dfs = folds_union(folds[:k_train]), folds_union(folds[-k_test:])
            naive_bayes = NaiveBayesMod(best_alpha, best_w)
            train_Xs, train_ys = dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                       for type_, df in train_dfs.items()]), \
                                 list(train_dfs.values())[0][c.field_class_].to_list()
            naive_bayes.fit(train_Xs, train_ys)
            naive_ys, true_ys = [naive_bayes.classify(X)
                                 for X in to_list_of_dicts_of_series(
                    dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                          for type_, df in test_dfs.items()]))], \
                                list(test_dfs.values())[0][c.field_class_].to_list()
            # Результаты по отложенной выборке
            metrics = calc_metrics(naive_ys, true_ys)
            vol_metric_list[train_vol].append(metrics)

# 1. Полный перебор
# Результаты по итогам стохастической валидации по отложенной выборке
vol_ms = {}
for train_vol, metric_list in vol_metric_list.items():
    vol_ms[train_vol] = calc_ms(metric_list)

with open(c.vol_ms_json_null_coef, 'w', encoding=c.encoding) as f:
    f.write(json.dumps(vol_ms))





