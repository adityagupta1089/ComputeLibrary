import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import itertools

with open("resnet50_TL80000_dt10000_temp_profile.txt") as f:
    content = f.read()
vals = re.findall(
    r"(\w+): start_temp = (\d+), index = \d+, max_temp = (\d+)", content
)
vals2 = {
    "CPU_SMALL": {65000: 4968, 70000: 1416, 75000: 3624, 80000: 1728},
    "CPU_BIG": {65000: 18144, 75000: 11088, 80000: 8112, 85000: 4998.86},
    "CPU_BIG_CPU_SMALL": {65000: 23328, 80000: 8070.55, 85000: 5481},
    "GPU": {65000: 16459.2, 70000: 11324.6, 80000: 3888},
    "GPU_CPU_SMALL": {65000: 20088, 70000: 11760, 75000: 11784, 80000: 7776},
    "GPU_CPU_BIG": {65000: 24624, 70000: 18360, 75000: 13024.8, 80000: 11880},
    "GPU_CPU_BIG_CPU_SMALL": {65000: 25272, 70000: 17403.4, 75000: 10998},
}

r = 2
c = 4
fig, axs = plt.subplots(r, c, figsize=(12, 8))
axs = axs.reshape(r * c)
plt.set_cmap("Dark2")
colors = plt.get_cmap().colors

data = defaultdict(list)
for graph, start_temp, max_temp in vals:
    data[graph].append((int(start_temp), int(max_temp)))

# actual values
for idx, (graph, val) in enumerate(data.items()):
    x, y = zip(*val)
    axs[idx].scatter(x, y, s=5, color=colors[idx], label=graph)

# avreage values
for idx, (graph, val) in enumerate(vals2.items()):
    x, y = zip(*sorted(val.items(), key=lambda kv: kv[0]))
    axs[idx].plot(x, y, "o-", color=colors[idx], label=graph)

coeffs = {}

# interpolated
for idx, (graph, val) in enumerate(vals2.items()):
    x, y = zip(*sorted(val.items(), key=lambda kv: kv[0]))
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    coeffs[graph] = coef
    axs[idx].plot(x, poly1d_fn(x), "--", color=colors[idx], label=graph)
    axs[-1].plot(x, poly1d_fn(x), "x-", color=colors[idx], label=graph)

for ax, graph in zip(axs[:-1], vals2.keys()):
    ax.set_xlabel(r"$T_0$")
    ax.set_ylabel(r"$\Delta T_{\rm max}$")
    ax.set_title(
        f"Graph {graph}\n $\Delta T_{{\\rm max}}={coeffs[graph][0]:.2f}T_0+{coeffs[graph][1]:.2f}$",
        fontsize=8,
    )
    ax.grid(which="both", axis="both")

axs[-1].set_title("All graphs (interpolated)")
axs[-1].grid(which="both", axis="both")
axs[-1].set_xlabel(r"$T_0$")
axs[-1].set_ylabel(r"$\Delta T_{\rm max}$")
handles = []
labels = []
for idx, graph in enumerate(vals2.keys()):
    handles.append(Patch(facecolor=colors[idx]))
    labels.append(graph)
handles += [
    Line2D([0], [0], linestyle="--", color="k"),
    Line2D([0], [0], linestyle="-", color="k", marker="o"),
    Line2D([0], [0], marker="o", linestyle="-", color="w", markerfacecolor="k"),
    Line2D([0], [0], linestyle="-", marker="x", color="k"),
]

labels += [
    "Interpolated",
    "Rounded to nearest 5000",
    "Actual Values (scatter plot)",
    "Interpolated (last figure)",
]


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


ncol = 5
fig.legend(
    flip(handles, ncol), flip(labels, ncol), ncol=ncol, loc="lower center"
)

plt.tight_layout()
fig.subplots_adjust(top=0.88, bottom=0.15)
fig.suptitle("Scheduler v3.1 Temperature Profiling (TL=80000, dt=10000)")
# fig.savefig("resnet50_TL80000_dt10000.png")
# plt.show()

for kv in coeffs.items():
    print(f"{{ {kv[0]}, {{ {kv[1][0]:.2f}, {kv[1][1]:.2f} }} }}")
