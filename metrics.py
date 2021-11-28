import numpy as np


def calc_metrics(classifier_ys, true_ys):
    rate = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
    }
    for classifier_y, true_y in zip(classifier_ys, true_ys):
        if classifier_y == true_y:
            if classifier_y == 1:
                rate['TP'] += 1
            else:
                rate['TN'] += 1
        else:
            if classifier_y == 1:
                rate['FP'] += 1
            else:
                rate['FN'] += 1

    acc = (rate['TP'] + rate['TN']) / len(classifier_ys)
    pre = rate['TP'] / (rate['TP'] + rate['FP']) if rate['TP'] != 0 else 0
    rec = rate['TP'] / (rate['TP'] + rate['FN']) if rate['TP'] != 0 else 0
    f = (2 * pre * rec) / (pre + rec) if pre + rec != 0 else 0

    return {
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'F': f
    }


def calc_ms(metrics_list):
    ms = {}
    for name in metrics_list[0].keys():
        metric_track = [metrics[name] for metrics in metrics_list]
        ms[name] = {
            'min': np.min(metric_track),
            'max': np.max(metric_track),
            'mean': np.mean(metric_track),
            'median': np.median(metric_track)
        }
    return ms





