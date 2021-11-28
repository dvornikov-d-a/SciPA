import matplotlib.pyplot as plt
import json
import config as c


with open(c.vol_ms_json, 'r', encoding=c.encoding) as f:
    vol_ms = json.loads(f.read())

xs = [int(vol) for vol in list(vol_ms.keys())]
ys = {}
for vol, ms in vol_ms.items():
    for metric_name, metric_ms in ms.items():
        for ms_name, point in metric_ms.items():
            if ms_name not in ys.keys():
                ys[ms_name] = {}
            if metric_name not in ys[ms_name].keys():
                ys[ms_name][metric_name] = []
            ys[ms_name][metric_name].append(point)

colors = ('r', 'g', 'b', 'k')
for ms_name, metrics in ys.items():
    fig, ax = plt.subplots()
    plt.xlabel('train dataset volume')
    plt.ylabel(f'{ms_name} values')
    for i, (metric_name, points) in enumerate(metrics.items()):
        plt.plot(xs, points, colors[i], label=metric_name)
    ax.legend()
    # plt.show()
    plt.savefig(f'res/plots/whole_{ms_name}s.png')

# for max_vol_i in [int(vol / c.min_vol) for vol in (150, 100, 50, 25)]:
#     for ms_name, metrics in ys.items():
#         fig, ax = plt.subplots()
#         plt.xlabel('train dataset volume')
#         plt.ylabel(f'{ms_name} values')
#         for i, (metric_name, points) in enumerate(metrics.items()):
#             plt.plot(xs[-max_vol_i:], points[-max_vol_i:], colors[i], label=metric_name)
#         ax.legend()
#         # plt.show()
#         plt.savefig(f'res/plots/max_vol_{max_vol_i * c.min_vol}_{ms_name}s.png')

