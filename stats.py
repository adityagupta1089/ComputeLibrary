import glob
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

temp_csvs = glob.glob("temp_schedulerv*/*_TL*_dt*.csv")
time_logs = glob.glob("temp_scheduler_all/*_TL*_dt*_run_sched*.log")

graphs = ("alexnet", "googlenet", "mobilenet", "resnet50", "squeezenet")
versions = ("3", "3.1", "4")
TLs = ("80000", "999999")


def dd():
    return defaultdict(dd)


stats = defaultdict(dd)

for temp_csv in temp_csvs:
    version, graph, TL = re.findall(
        r"temp_schedulerv([\d_]+)/(\w+)_TL(\d+)_dt\d+.csv", temp_csv
    )[0]
    version = version.replace("_", ".")
    df = pd.read_csv(temp_csv)
    average_temp = df[" temp"].mean()
    crosses_threshold = len(df[df[" temp"] > int(TL)]) / len(df[" temp"]) * 100
    stats[graph][version][TL]["average_temp"] = average_temp
    stats[graph][version][TL]["crosses_threshold"] = crosses_threshold

for time_log in time_logs:
    file_name = time_log.split("/")[1]
    graph, TL, version = re.findall(
        r"(\w+)_TL(\d+)_dt\d+_run_sched([\d\.]+).log", file_name
    )[0]
    with open(time_log) as f:
        content = f.read()
    time = re.findall(r"([\d\.]+) per inference", content)[0]
    stats[graph][version][TL]["time_taken"] = time

indexes = []
rows = []
for graph in sorted(stats):
    for version in sorted(stats[graph]):
        if version not in versions:
            continue
        for TL in sorted(stats[graph][version]):
            if TL not in TLs:
                continue
            data = stats[graph][version][TL]
            indexes.append((graph, version, TL,))
            rows.append(
                [
                    float(data["average_temp"]),
                    float(data["crosses_threshold"]),
                    float(data["time_taken"]),
                ]
            )
df = pd.DataFrame(
    rows,
    columns=["Average Temp", "Crosses Threshold", "Time Taken",],
    index=pd.MultiIndex.from_tuples(indexes, names=["Graph", "Version", "TL"]),
)

matplotlib.style.use("bmh")
tab20 = matplotlib.cm.get_cmap("tab20")
poss = np.linspace(0, 1, 20)[:6]
colors = [tab20(x) for x in poss]
for column in df.columns:
    fig = plt.figure(figsize=(12, 8))
    ax: Axes = df.loc[:, column].plot(kind="bar", color=colors, stacked=True)
    ax.legend(
        [Line2D([0], [0], color=col, lw=4) for x, col in zip(poss, colors)],
        [f"v{version}, TL={TL}" for version in versions for TL in TLs],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    m, M = ax.get_xlim()
    xticks = np.linspace(m, M, len(graphs) + 1)
    ax.set_xticks([(x + x1) / 2 for x, x1 in zip(xticks, xticks[1:])])
    ax.set_xticklabels(graphs, rotation=0)
    for x in np.linspace(m, M, len(graphs) + 1)[1:-1]:
        ax.axvline(x=x)
    ax.set_ylabel(column)
    ax.set_xlabel("Graph")
    plt.title(f"{column} vs. Graph")
    fig.subplots_adjust(right=0.82)
    # plt.show()
    fig.savefig(f"temp_scheduler_{column.lower().replace(' ', '_')}.png")
