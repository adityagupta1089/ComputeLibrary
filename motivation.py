import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use("bmh")

data = {
    "AlexNet": [
        2.04546,
        0.204546,
        10,
        0,
        1.68578,
        0.168578,
        0,
        10,
        1.08566,
        0.108566,
        5,
        5,
    ],
    "MobileNet": [
        0.8054,
        0.08054,
        10,
        0,
        0.847036,
        0.0847036,
        0,
        10,
        0.597024,
        0.0597024,
        6,
        4,
    ],
    "GoogleNet": [
        1.74059,
        0.174059,
        10,
        0,
        3.32846,
        0.332846,
        0,
        10,
        1.4559,
        0.14559,
        7,
        3,
    ],
    "SqueezeNet": [
        0.903475,
        0.0903475,
        10,
        0,
        1.27485,
        0.127485,
        0,
        10,
        0.876682,
        0.0876682,
        7,
        3,
    ],
    "ResNet50": [
        4.28826,
        0.428826,
        10,
        0,
        3.94294,
        0.394294,
        0,
        10,
        2.65938,
        0.265938,
        4,
        6,
    ],
}
rows = []
for graph, vals in data.items():
    rows += (
        [graph, "CPU", vals[1]],
        [graph, "GPU", vals[5]],
        [graph, "CPU + GPU", vals[9]],
    )

df = pd.DataFrame(rows, columns=["Graph", "Target", "Time (sec/image)"])

sns.catplot(
    data=df,
    x="Graph",
    y="Time (sec/image)",
    hue="Target",
    legend=True,
    kind="bar",
    legend_out=False,
)
print("Generated fig")
plt.savefig("motivation.png")

