import glob
import re
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

matplotlib.style.use("bmh")
tab20 = matplotlib.cm.get_cmap("tab20")
poss = np.linspace(0, 1, 20)[:6]
colors = [tab20(x) for x in poss]

temp_csvs = glob.glob("temp_schedulerv*/*_TL*_dt10000.csv")
time_logs = glob.glob("temp_scheduler_all/*_TL*_dt10000_run_sched*.log")

graphs = ("alexnet", "googlenet", "mobilenet", "resnet50", "squeezenet")
versions = ("3", "3.1", "4")
TLs = ("80000", "999999")


def dd():
    return defaultdict(dd)


def map_tl(tl):
    if int(tl) == 80000:
        return "80^\circ"
    elif int(tl) == 999999:
        return "\infty"


stats = defaultdict(dd)
temp_df = pd.DataFrame()

for temp_csv in temp_csvs:
    version, graph, TL = re.findall(
        r"temp_schedulerv([\d_]+)/(\w+)_TL(\d+)_dt10000.csv", temp_csv
    )[0]
    version = version.replace("_", ".")
    if TL not in TLs or version not in versions:
        continue
    df = pd.read_csv(temp_csv)
    average_temp = df["temp"].mean()
    crosses_threshold = len(df[df["temp"] > int(TL)]) / len(df["temp"]) * 100
    stats[graph][version][TL]["crosses_threshold"] = crosses_threshold
    df["temp"] /= 1000
    stats[graph][version][TL]["temps"] = df
    temp_df[f"{graph}_v{version}_TL{TL}"] = df["temp"]

for time_log in time_logs:
    file_name = time_log.split("/")[1]
    graph, TL, version = re.findall(
        r"(\w+)_TL(\d+)_dt\d+_run_sched([\d\.]+).log", file_name
    )[0]
    with open(time_log) as f:
        content = f.read()
    time = re.findall(r"([\d\.]+) per inference", content)[0]
    stats[graph][version][TL]["time_taken"] = time

print(temp_df)
xtick_lables = []
required_columns = []
for graph in sorted(stats):
    for version in sorted(stats[graph]):
        for TL in sorted(stats[graph][version]):
            xtick_lables.append(graph)
            fig = plt.figure()
            sns.lineplot(data=stats[graph][version][TL]["temps"], x="time", y="temp")
            if int(TL) == 80000:
                plt.axhline(80, linestyle="--", color="r")
            plt.xlabel("Time (sec)")
            plt.ylabel("Temperature $(^\circ C)$")
            plt.title(
                f"Execution scheduler {graph}, v{version}, ${{\\rm TL}} = {map_tl(TL)}$"
            )
            plt.show()
            fig.savefig(
                f"temp_schedulerv{version.replace('.', '_')}/{graph}_TL{TL}.png"
            )
            required_columns.append(f"{graph}_v{version}_TL{TL}")
temp_df = temp_df[required_columns]
fig = plt.figure(figsize=(12, 8))
ax = sns.boxplot(
    data=temp_df,
    dodge=True,
    palette=sns.color_palette("tab20", n_colors=6),
    showfliers=False,
    linewidth=1,
)
ax.set_xlabel("Graph")
m, M = ax.get_xlim()
xtick_lables = np.linspace(m, M, len(graphs) + 1)
ax.set_xticks([(x + x1) / 2 for x, x1 in zip(xtick_lables, xtick_lables[1:])])
ax.set_xticklabels(graphs, rotation=0)
for x in np.linspace(m, M, len(graphs) + 1)[1:-1]:
    ax.axvline(x=x, linewidth=1)
ax.axhline(y=80, color="r", alpha=0.5)
ax.legend(
    [Line2D([0], [0], color=col, lw=4) for x, col in zip(poss, colors)],
    [f"v{version}, ${{\\rm TL}}={map_tl(TL)}$" for version in versions for TL in TLs],
    loc="center left",
    bbox_to_anchor=(1, 0.3),
)
ax.set_ylabel(r"Temperature $(^\circ C)$")
plt.title("Temperature vs. Graph")
plt.show()
fig.savefig("temp_scheduler_temperatures.png")

indexes = []
rows = []
for graph in sorted(stats):
    for version in sorted(stats[graph]):
        for TL in sorted(stats[graph][version]):
            data = stats[graph][version][TL]
            indexes.append((graph, version, float(TL) / 1000))
            rows.append(
                [float(data["crosses_threshold"]), float(data["time_taken"]),]
            )


df = pd.DataFrame(
    rows,
    columns=["Crosses Threshold", "Time Taken"],
    index=pd.MultiIndex.from_tuples(indexes, names=["Graph", "Version", "TL"]),
)
print(df)

for column in df.columns:
    fig = plt.figure(figsize=(12, 8))
    ax: Axes = df.loc[:, column].plot(kind="bar", color=colors, stacked=True)
    ax.legend(
        [Line2D([0], [0], color=col, lw=4) for x, col in zip(poss, colors)],
        [
            f"v{version}, ${{\\rm TL}}={map_tl(TL)}$"
            for version in versions
            for TL in TLs
        ],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    m, M = ax.get_xlim()
    xtick_lables = np.linspace(m, M, len(graphs) + 1)
    ax.set_xticks([(x + x1) / 2 for x, x1 in zip(xtick_lables, xtick_lables[1:])])
    ax.set_xticklabels(graphs, rotation=0)
    for x in np.linspace(m, M, len(graphs) + 1)[1:-1]:
        ax.axvline(x=x)
    if column == "Crosses Threshold":
        ax.set_ylabel(column + " (%)")
    elif column == "Time Taken":
        ax.set_ylabel(column + " (sec)")
    ax.set_xlabel("Graph")
    plt.title(f"{column} vs. Graph")
    fig.subplots_adjust(right=0.82)
    plt.show()
    fig.savefig(f"temp_scheduler_{column.lower().replace(' ', '_')}.png")
