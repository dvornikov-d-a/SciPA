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


# Варианты альф
alphas = (1, 0.8, 0.5)

# Варианты весов
ws = (
    # Слова, авторы, журналы, области
    frozendict(zip(c.set_names, (1, 1, 1, 1))),
    frozendict(zip(c.set_names, (1, 4, 3, 2)))
)

alphas_weights = [(alpha, w) for alpha in alphas for w in ws]

bests_i_metrics = []

dfs = dict(zip([s for s in c.set_names], [pd.read_csv(f'{c.data_prev}{s}.csv') for s in c.set_names]))
for shuffle_number in range(c.shuffle_count):
    dfs_shuffled_balanced = do_balance(shuffle(dfs, random_state=shuffle_number), c.field_class_, c.classes)
    folds = k_folds_split(dfs_shuffled_balanced, c.k)
    for control_dfs_index, control_dfs in enumerate(folds):
        folds_but_control = deepcopy(folds)
        folds_but_control.pop(control_dfs_index)
        cv_res = []
        for alpha, w in alphas_weights:
            fs = []
            for test_dfs_index, test_dfs in enumerate(folds_but_control):
                folds_but_test = deepcopy(folds_but_control)
                folds_but_test.pop(test_dfs_index)
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
        cv_best_params_i = np.argmax(cv_res)
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
        control_res = (cv_best_params_i, metrics)
        bests_i_metrics.append(control_res)

paramss_mss = []
for param_i in set([param_i for param_i, metrics in bests_i_metrics]):
    ms = calc_ms([m for p_i, m in bests_i_metrics if param_i == p_i])
    alpha = alphas_weights[param_i][0]
    w = tuple(alphas_weights[param_i][1].values())
    paramss_mss.append({'alpha': alpha, 'weights': w, 'ms': ms})
with open(c.best_params_ms_json, 'w', encoding=c.encoding) as f:
    f.write(json.dumps(paramss_mss))

best_params = alphas_weights[bests_i_metrics[np.argmax([metrics['F'] for params_i, metrics in bests_i_metrics])][0]]
with open(c.best_params_txt, 'w', encoding=c.encoding) as f:
    f.write(f'alpha: {best_params[0]}\n'
            f'weights: {tuple(best_params[1].values())}')










