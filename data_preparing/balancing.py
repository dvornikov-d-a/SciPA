import math


# Функция вычисления соотношения объектов разных классов
def ratio(df, class_col, classes):
    class_percents = {}
    for i in classes:
        class_percents[i] = round(df[class_col].to_list().count(i) * 100 / len(df), 2)
    return class_percents


# Функция сотворения баланса двух классов
def do_balance(dfs, class_col, classes):
    class_percents = ratio(dfs[0], class_col, classes)
    max_class = {'class': 0, 'percent': 0}
    min_class = {'class': 0, 'percent': 100}
    for class_, percent in class_percents.items():
        if percent > max_class['percent']:
            max_class['class'] = class_
            max_class['percent'] = percent
        elif percent < min_class['percent']:
            min_class['class'] = class_
            min_class['percent'] = percent
    diff = {'percent': round(max_class['percent'] - min_class['percent'], 2)}
    diff['count'] = math.floor(len(dfs[0]) * diff['percent'] / 100)
    if diff['percent'] < 5:
        return dfs
    else:
        indexes = []
        for i, class_ in dfs[0]['class'].items():
            if diff['count'] > 0:
                if class_ == max_class['class']:
                    indexes.append(i)
                    diff['count'] -= 1
            else:
                break
        new_dfs = []
        for df in dfs:
            new_dfs.append(df.drop(labels=indexes, axis='index'))
        return new_dfs
