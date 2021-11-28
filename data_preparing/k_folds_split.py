import math
import pandas as pd


def k_folds_split(dfs, k):
    n = len(list(dfs.values())[0])
    fold_volume = math.floor(n / k)

    folds = []
    for i in range(k):
        fold = {}
        for type_, df in dfs.items():
            fold[type_] = df.loc[i * fold_volume:(i + 1) * fold_volume]
        folds.append(fold)
    return folds


def folds_union(folds):
    dfs = dict(zip([type_ for type_ in folds[0].keys()],
                   [pd.DataFrame(columns=df.columns) for df in folds[0].values()]))
    for fold in folds:
        for type_, df in fold.items():
            dfs[type_] = dfs[type_].append(df, ignore_index=True)
    return dfs

