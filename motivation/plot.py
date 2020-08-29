import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

matplotlib.style.use("bmh")

for file in os.listdir("."):
    if "resnet50.csv" in file:
        graph_name = file.split(".")[0]
        print(graph_name)
        df = pd.read_csv(file, names=["time", "temp"],)
        df["temp"] /= 1000
        ax = df.plot(x="time", y="temp", label="Temperature")
        df.rolling(round(20 / 0.01)).mean().plot(
            x="time", y="temp", ax=ax, label="Moving average in 20 seconds"
        )
        ax.xaxis.grid(True, which="both")
        ax.yaxis.grid(True, which="both")
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Temperature $(^\circ C)$")
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        plt.title(f"600 batches of batch size 10 on {graph_name}")
        plt.savefig(f"{graph_name}.png")
