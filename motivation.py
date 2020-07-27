import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.style.use("bmh")

data = {
    "AlexNet": """
CPU
2.04546, 0.204546, 10, 0
8.20984, 0.0820984, 10, 0
1.48004, 1.48004, 100, 0
GPU
1.68578, 0.168578, 0, 10
3.88249, 0.0388249, 0, 10
1.02315, 1.02315, 0, 100
CPU+GPU
1.08566, 0.108566, 5, 5
4.02829, 0.0402829, 3, 7
0.565674, 0.565674, 36, 64
""",
    "MobileNet": """
CPU
0.8054, 0.08054, 10, 0
7.26474, 0.0726474, 10, 0
0.165057, 0.165057, 100, 0
GPU
0.847036, 0.0847036, 0, 10
4.07821, 0.0407821, 0, 10
0.349245, 0.349245, 0, 100
CPU+GPU
0.597024, 0.0597024, 6, 4
3.06056, 0.0306056, 3, 7
0.121253, 0.121253, 70, 30
""",
    "GoogleNet": """
CPU
1.74059, 0.174059, 10, 0
16.394, 0.16394, 10, 0
0.40172, 0.40172, 100, 0
GPU
3.32846, 0.332846, 0, 10
8.56587, 0.0856587, 0, 10
0.908863, 0.908863, 0, 100
CPU+GPU
1.4559, 0.14559, 7, 3
5.32265, 0.0532265, 2, 8
0.292517, 0.292517, 70, 30
""",
    "SqueezeNet": """
CPU
0.903475, 0.0903475, 10, 0
8.74938, 0.0874938, 10, 0
0.154127, 0.154127, 100, 0
GPU
1.27485, 0.127485, 0, 10
4.3122, 0.043122, 0, 10
0.354992, 0.354992, 0, 100
CPU+GPU
0.876682, 0.0876682, 7, 3
3.23579, 0.0323579, 2, 8
0.12756, 0.12756, 79, 21
""",
    "ResNet50": """
CPU
4.28826, 0.428826, 10, 0 
30.681, 0.30681, 10, 0
2.32899, 2.32899, 100, 0
GPU
3.94294, 0.394294, 0, 10
18.3556, 0.183556, 0, 10
1.94383, 1.94383, 100, 0
CPU+GPU
2.65938, 0.265938, 4, 6
12.9827, 0.129827, 2, 8
1.17527, 1.17527, 42, 58
""",
}

data2 = []
rows = [1, 2, 3, 5, 6, 7, 9, 10, 11]
rowvs = [
    (target, config)
    for target in ["CPU", "GPU", "CPU + GPU"]
    for config in [(10, 10), (10, 100), (1, 100)]
]
data3 = defaultdict(list)
for graph, val in data.items():
    print(graph)
    for (target, config), (idx, line) in zip(
        rowvs,
        [
            (a, b)
            for a, b in enumerate([x for x in val.split("\n") if x])
            if a in rows
        ],
    ):
        data3[config].append([graph, target, line.split(",")[1]])

for config in data3:
    df = pd.DataFrame(
        data3[config], columns=["Graph", "Target", "Time (sec/inference)",],
    )
    print(df)

    sns.catplot(
        data=df,
        x="Graph",
        y="Time (sec/inference)",
        hue="Target",
        legend=True,
        kind="bar",
        legend_out=False,
    )
    plt.title(f"{config[0]} Batches, {config[1]} Batch Size")
    print(f"Generated fig {config}")
    plt.savefig(f"motivation_{config[0]}_{config[1]}.png")
