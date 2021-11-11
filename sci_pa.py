import pandas as pd

from data_preparing.balancing import do_balance
from naive_bayes import NaiveBayes

# df = pd.read_csv('src/papers_structed.csv')
# df_balanced = Balancer().do_balance(df, 'class', [0, 1])
# train_dataset_volumes = [200]  # , 100, 50, 25, 20, 15, 10, 5]

# learning_model = NaiveBayes()
# validation_mod = Validation(learning_model, df_balanced, train_dataset_volumes, shuffle_count=1)
# validation_mod.validate()
# for train_dataset_volume in train_dataset_volumes:
#     print(f'Размер обучающей выборки: {train_dataset_volume}.')
#     for agreg in ['min', 'mean', 'median', 'max']:
#         for metric in learning_model.metrics:
#             print(f'{agreg}-{metric}: {validation_mod.res[train_dataset_volume][agreg][metric]}')
#         print()
#     print('---------------------------------------')

# df = pd.read_csv('messages.csv')
# learning_model = NaiveBayes()
# learning_model.fit(df)
# class_ = learning_model.classify({'магазин': 1, 'гора': 1, 'яблоко': 1, 'купи': 1, 'семь': 1, 'килограмм': 1, 'шоколадка': 1})
# print(class_)

df = pd.read_csv('src/papers_structed.csv')
df_balanced = do_balance(df, 'class', [0, 1])
df_balanced_shuffled = df_balanced.sample(frac=1).reset_index(drop=True)
train_dataset, test_dataset = df_balanced_shuffled[:200], df_balanced_shuffled[200:]
learning_model = NaiveBayes()
learning_model.fit(train_dataset)
test_Xs = test_dataset.drop(labels=['id', 'class'], axis='columns').to_dict('records')
test_true_ys = test_dataset['class'].to_list()
test_naive_ys = []
for X in test_Xs:
    ans = learning_model.classify(X)
    test_naive_ys.append(1 if ans[1] > ans[0] else 0)
rate = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
for naive, true in zip(test_naive_ys, test_true_ys):
    if naive == true:
        if naive == 1:
            rate['TP'] += 1
        else:
            rate['TN'] += 1
    else:
        if naive == 1:
            rate['FP'] += 1
        else:
            rate['FN'] += 1
accuracy = (rate['TP'] + rate['TN']) / len(test_naive_ys)
precision = rate['TP'] / (rate['TP'] + rate['FP'])
recall = rate['TP'] / (rate['TP'] + rate['FN'])
print('accuracy:\t', accuracy)
print('precision:\t', precision)
print('recall:\t\t', recall)


