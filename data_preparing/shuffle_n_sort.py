import random as rnd
import pandas as pd


# Функция перемешивания нескольких связанных по индексам датафреймов
def shuffle_n_sort(dfs):
    n = len(list(dfs.values())[0])
    indices = [i for i in range(n)]
    rnd.shuffle(indices)

    new_dfs = {}
    for type_, df in dfs.items():
        new_df = pd.DataFrame(columns=df.columns)
        for i in indices:
            new_df.loc[len(new_df)] = df.iloc[i]
            # new_df = new_df.append(df.iloc[i])
        new_dfs[type_] = new_df
    return new_dfs


def shuffle(dfs, random_state=1):
    new_dfs = {}
    for type_, df in dfs.items():
        new_dfs[type_] = df.sample(frac=1, random_state=random_state, ignore_index=True)
    return new_dfs

