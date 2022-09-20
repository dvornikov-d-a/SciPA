field_id_ = 'id_'
field_words_ = 'words_'
field_authors_ = 'authors_'
field_journals_ = 'journals_'
field_fields_ = 'fields_'
field_class_ = 'class_'

classes = (0, 1)

set_names = (field_words_, field_authors_, field_journals_, field_fields_)

import math
import numpy as np
import config as c
from frozendict import frozendict

class NaiveBayesMod:
    def __init__(self, alpha=0.8,
                 l_w=frozendict(
                     {c.field_words_: 1,
                      c.field_authors_: 1,
                      c.field_journals_: 0,
                      c.field_fields_: 0}),
                 classes=c.classes):
        # Гиперпараметр
        self._alpha = alpha
        # Типы признаков
        # (str, ...)
        self._ls = tuple(l_w.keys())
        # Веса типов признаков, преобразованные из соотношения частей, не
        # превышающие 1
        # (float, ...)
        self._w = tuple([round(w / max(l_w.values()), 2) for w in l_w.values()])
        # Классы принадлежности документов
        self._classes = classes
        # Основные общие значения (для расчёта значений основной таблицы)
        # ['Q'][class_j] или ['M'][l_i] или ['N'][l_i][class_j]
        self._main_values = {
            'Q': dict(zip([j for j in range(len(self._classes))],
                          [0.0 for j in range(len(self._classes))])),
            'M': dict(zip([i for i in range(len(self._ls))],
                          [0 for i in range(len(self._ls))])),
            'N': dict(zip([i for i in range(len(self._ls))],
                          [dict(zip([j for j in range(len(self._classes))],
                                    [0 for j in range(len(self._classes))]))
                           for i in range(len(self._ls))]))
        }
        # Основные таблицы (ключевые вероятностные характеристики термов)
        # [l_i][term][class_j]['N'] или [l_i][term][class_j]['P']
        self._main_tables = tuple([
            {'_': dict(zip([j for j in range(len(self._classes))],
                           [{'N': 0, 'P': 0} for j in range(len(self._classes))]))}
            for i in range(len(self._ls))
        ])

        # Xs: dict of pandas.DataFrames, ys: list of ints

    def fit(self, Xs, ys):
        for j in range(len(self._classes)):
            self._main_values['Q'][j] = (sum([1 for y in ys
                                              if y == self._classes[j]]) +
                                         self._alpha) \
                                        / (len(ys) + self._alpha)

        for i, l in enumerate(self._ls):
            for term in Xs[l]:
                self._main_values['M'][i] += 1
                self._main_tables[i][term] = dict(zip([j for j in
                                                       range(len(self._classes))],
                                                      [{'N': 0, 'P': 0} for j in
                                                       range(len(self._classes))]))
                for count, j in zip(Xs[l][term], [self._classes.index(y)
                                                  for y in ys]):
                    # self._main_values['M'][i] += count
                    self._main_values['N'][i][j] += count
                    self._main_tables[i][term][j]['N'] += count

        for i in range(len(self._ls)):
            for term in self._main_tables[i].keys():
                for j in self._main_tables[i][term].keys():
                    self._main_tables[i][term][j]['P'] = self._P_term(i, term, j)

        # Функция классификации документа

    def classify(self, X):
        return self._classes[np.argmax([self._log_P(X, j, 10) for j
                                        in range(len(self._classes))])]

        # Функция расчёта вероятностей принадлежности классам

    def estimate(self, X):
        estimations = {}
        for class_index in range(len(self._classes)):
            estimations[class_index] = self._log_P(X, class_index, 10)
        return estimations

    # Функция расчёта логарифма вероятности принадлежности документа
    # определённому классу
    # X: dict of pandas.Series
    def _log_P(self, X, class_j, base):
        sum_ = math.log(self._main_values['Q'][class_j], base)
        for i, l in enumerate(self._ls):
            for term, count in X[l].items():
                if term in self._main_tables[i].keys():
                    sum_ += count * self._w[i] \
                            * math.log(self._main_tables[i][term][class_j]['P'])
                else:
                    sum_ += count * self._w[i] \
                            * math.log(self._main_tables[i]['_'][class_j]['P'])
        return sum_

    # Функция расчёта вероятности принадлежности терма статье определённого класса
    def _P_term(self, l_i, term, class_j):
        return (self._alpha + self._main_tables[l_i][term][class_j]['N']) \
               / (self._alpha * self._main_values['M'][l_i]
                  + self._main_values['N'][l_i][class_j])

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

def rank(dfs, naive_bayes):
    # Ранжирование выдачи заново (сортировка по убыванию)
        naive_estimations = [naive_bayes.estimate(X)
                             for id_, X in
                             zip(to_list_of_dicts_of_series(dict([(type_,
                                 df[c.field_id_]) for type_, df in dfs.items()])),
                                 to_list_of_dicts_of_series(dict([(type_,
                                 df.drop(columns=[c.field_id_, c.field_class_]))
                                 for type_, df in dfs.items()])))]
        scores = [naive_estimation[c.classes[1]] - naive_estimation[c.classes[0]]
                  for naive_estimation in naive_estimations]
        scores_ = 'scores_'
        for s in c.set_names:
            dfs[s][scores_] = scores
            dfs[s].sort_values(scores_, ascending=False)
            dfs[s].drop(scores_, axis=1, inplace=True)
