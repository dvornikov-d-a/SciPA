import json
import pandas as pd
import numpy as np
from metrics import calc_metrics, calc_ms
import config as c
from frozendict import frozendict
from copy import deepcopy
from naive_bayes_mod import NaiveBayesMod
from data_preparing.shuffle_n_sort import shuffle
from data_preparing.balancing import do_balance
from data_preparing.k_folds_split import k_folds_split
from data_preparing.k_folds_split import folds_union
from data_preparing.dict_list_worker import to_list_of_dicts_of_series
import time


# Варианты альф
alphas = (0.5,)

# Варианты весов
ws = (
    # Слова, авторы, журналы, области
    frozendict(zip(c.set_names, (1, 1, 1, 1))),
    frozendict(zip(c.set_names, (1, 0, 0, 0))),
    frozendict(zip(c.set_names, (1, 0, 0, 1))),
    frozendict(zip(c.set_names, (1, 0, 0, 4))),
    frozendict(zip(c.set_names, (1, 0, 1, 0))),
    frozendict(zip(c.set_names, (1, 0, 4, 0))),
    frozendict(zip(c.set_names, (1, 1, 0, 0))),
    frozendict(zip(c.set_names, (1, 4, 0, 0))),
)

alphas_weights = [(alpha, w) for alpha in alphas for w in ws]

# 0. Подбор гиперпараметров
# Результаты по итогам кросс-валидации с использованием стохастической валидации по отложенной выборке
bests_i_metrics = []

# Начало: 28.03.2022 17:30
shuffle_count = 100
# Конец:

dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))
# Несколько перемешиваний
for shuffle_number in range(shuffle_count):
    print(f'Progress: {shuffle_number}/{shuffle_count}')

    dfs_shuffled_balanced = do_balance(shuffle(dfs, random_state=shuffle_number), c.field_class_, c.classes)
    # Выделение k блоков для кросс-валидации (k-folds CV)
    folds = k_folds_split(dfs_shuffled_balanced, c.k)
    # Выделение контрольной выборки (Z) [1 блок] (отложенная выборка)
    for control_dfs_index, control_dfs in enumerate(folds):
        folds_but_control = deepcopy(folds)
        folds_but_control.pop(control_dfs_index)
        cv_res = []
        # Перебор возможных гиперпараметров алгоритма
        for alpha, w in alphas_weights:
            fs = []
            # Выделение тестовой выборки (Y) [1 блок, кроме контрольного]
            for test_dfs_index, test_dfs in enumerate(folds_but_control):
                folds_but_test = deepcopy(folds_but_control)
                folds_but_test.pop(test_dfs_index)
                # Выделение обучающей выборки (X) [оставшиеся блоки, кроме контрольного и тестового]
                train_dfs = folds_union(folds_but_control)
                naive_bayes = NaiveBayesMod(alpha, w)
                train_Xs, train_ys = dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                           for type_, df in train_dfs.items()]), \
                                     list(train_dfs.values())[0][c.field_class_].to_list()
                naive_bayes.fit(train_Xs, train_ys)
                naive_ys, true_ys = [naive_bayes.classify(X)
                                     for X in to_list_of_dicts_of_series(
                        dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                              for type_, df in test_dfs.items()]))], \
                                    list(test_dfs.values())[0][c.field_class_].to_list()
                metrics = calc_metrics(naive_ys, true_ys)
                fs.append(metrics['F'])
            f_median = np.median(fs)
            cv_res.append(f_median)
        # Выбор лучших гиперпараметров
        cv_best_params_i = np.argmax(cv_res)
        # Выделение обучающей выборки [все блоки, кроме контрольного]
        train_dfs = folds_union(folds_but_control)
        naive_bayes = NaiveBayesMod(*alphas_weights[cv_best_params_i])
        train_Xs, train_ys = dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                                   for type_, df in train_dfs.items()]), \
                             list(train_dfs.values())[0][c.field_class_].to_list()
        naive_bayes.fit(train_Xs, train_ys)
        naive_ys, true_ys = [naive_bayes.classify(X)
                             for X in to_list_of_dicts_of_series(
                dict([(type_, df.drop(columns=[c.field_id_, c.field_class_]))
                      for type_, df in control_dfs.items()]))], \
                            list(control_dfs.values())[0][c.field_class_].to_list()
        metrics = calc_metrics(naive_ys, true_ys)
        # Результаты валидации на отложенной выборке (Z)
        control_res = (cv_best_params_i, metrics)
        bests_i_metrics.append(control_res)

paramss_mss = []
for param_i in set([param_i for param_i, metrics in bests_i_metrics]):
    ms = calc_ms([m for p_i, m in bests_i_metrics if param_i == p_i])
    alpha = alphas_weights[param_i][0]
    w = tuple(alphas_weights[param_i][1].values())
    paramss_mss.append({'alpha': alpha, 'weights': w, 'ms': ms})
with open('res/best_params_null_coef_ms.json', 'w', encoding=c.encoding) as f:
    f.write(json.dumps(paramss_mss))

# best_params = alphas_weights[bests_i_metrics[np.argmax([metrics['F'] for params_i, metrics in bests_i_metrics])][0]]
# with open(c.best_params_txt, 'w', encoding=c.encoding) as f:
#     f.write(f'alpha: {best_params[0]}\n'
#             f'weights: {tuple(best_params[1].values())}')










