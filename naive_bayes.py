import math


class NaiveBayes:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.M = 0
        self.common_count = {
            'Q_1': 0,
            'Q_0': 0
        }
        self.N = {
            '_1': 0,
            '_0': 0
        }
        self.terms = None
        self._funcs = {
            'P(Q_k)': lambda k: (self.common_count[f'Q_{k}'] + self.alpha/10) /
                                ((self.common_count['Q_1'] + self.common_count['Q_0']) * (1 + self.alpha/10)),
            'P(x_j|Q_k)': lambda x_j, k: (self.alpha + self.terms[x_j][f'N_j{k}']) / (
                        self.alpha * self.M + self.N[f'_{k}'])
        }
        self._fited = False
        self.scores = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0
        }

    def _clear(self):
        self.M = 0
        self.common_count = {
            'Q_1': 0,
            'Q_0': 0
        }
        self.N = {
            '_1': 0,
            '_0': 0
        }
        self.terms = None
        self._fited = False
        self.scores = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0
        }

    @property
    def metrics(self):
        return tuple(self.scores.keys())

    @staticmethod
    def filter_features(dataset):
        columns_to_drop = []
        for col in dataset.drop(labels=['id', 'class'], axis='columns'):
            no_data = True
            for i, v in dataset[col].items():
                if v > 0:
                    no_data = False
                    break
            if no_data:
                columns_to_drop.append(col)
        dataset.drop(labels=columns_to_drop, axis='columns', inplace=True)

    def P(self, X_i, Q_k):
        multi = 1
        for x_j in X_i.keys():
            self._check_insert_term(x_j)
            multi *= self.terms[x_j][f'P(x_j|{Q_k})']
        return self._funcs['P(Q_k)'](Q_k) * multi

    def logP(self, X_i, Q_k, base):
        sum_ = 0
        for x_j, count_j in X_i.items():
            self._check_insert_term(x_j)
            sum_ += count_j * math.log(self.terms[x_j][f'P(x_j|{Q_k})'], base)
        return math.log(self._funcs['P(Q_k)'](Q_k), base) + sum_

    def _check_insert_term(self, term):
        if term not in self.terms.keys():
            self.terms[term] = {
                'N_j1': 0,
                'N_j0': 0
            }
            for k in [1, 0]:
                self.terms[term][f'P(x_j|{k})'] = self._funcs['P(x_j|Q_k)'](term, k)


    def fit(self, train_dataset):
        if self._fited:
            self._clear()
        ids, terms, Xs, ys = self._mine(train_dataset)
        self._pre_train(terms, ys)
        self._train(Xs, ys)
        self._fited = True

    def test(self, test_dataset):
        if not self._fited:
            print('Алгоритм не обучен!')
            return
        ids, terms, Xs, ys = self._mine(test_dataset)
        for term, params in terms.items():
            if term not in self.terms.keys():
                self.terms[term] = params
                for k in [1, 0]:
                    self.terms[term][f'P(x_j|{k})'] = self._funcs['P(x_j|Q_k)'](term, k)
        naive_ys = [self.classify(X) for X in Xs]
        self.calc_scores(naive_ys, ys)

    def calc_scores(self, naive_ys, true_ys):
        rates = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0
        }
        for naive_y, true_y in zip(naive_ys, true_ys):
            if true_y == naive_y:
                if naive_y == 1:
                    rates['TP'] += 1
                else:
                    rates['TN'] += 1
            else:
                if naive_y == 1:
                    rates['FP'] += 1
                else:
                    rates['FN'] += 1
        self.scores = {
            'accuracy': (rates['TP'] + rates['TN']) / len(naive_ys) if len(naive_ys) > 0 else 0,
            'precision': rates['TP'] / (rates['TP'] + rates['FP']) if rates['TP'] + rates['FP'] > 0 else 0,
            'recall': rates['TP'] / (rates['TP'] + rates['FN']) if rates['TP'] + rates['FN'] > 0 else 0
        }

    def classify(self, X):
        return {1: self.logP(X, 1, 2), 0: self.logP(X, 0, 2)}

    def _mine(self, dataset):
        self.filter_features(dataset)
        # [..., 'id_i', ...]
        ids = dataset['id'].to_list()
        terms = {}
        for term in dataset.drop(labels=['id', 'class'], axis='columns'):
            terms[term] = {
                'N_j1': 0,
                'N_j0': 0,
                'P(x_j|1)': 0,
                'P(x_j|0)': 0
            }
        # [..., {..., term: count, ...}, ...]
        Xs = dataset.drop(labels=['id', 'class'], axis='columns').to_dict('records')
        # [..., 0/1, ...]
        ys = dataset['class'].to_list()
        return ids, terms, Xs, ys

    def _pre_train(self, terms, ys):
        self.M = len(terms.keys())
        self.terms = terms
        for k in [1, 0]:
            self.common_count[f'Q_{k}'] = len([y for y in ys if y == k])

    def _train(self, Xs, ys):
        for X, y in zip(Xs, ys):
            for term, count in X.items():
                self.terms[term][f'N_j{y}'] += count
                self.N[f'_{y}'] += count
        for term, params in self.terms.items():
            for k in [1, 0]:
                params[f'P(x_j|{k})'] = self._funcs['P(x_j|Q_k)'](term, k)