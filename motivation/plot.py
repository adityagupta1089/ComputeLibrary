import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use("bmh")

for file in os.listdir("."):
    if ".csv" in file:
        graph_name = file.split(".")[0]
        print(graph_name)
        df = pd.read_csv(file, names=["time", "temp"])
        df["temp"] /= 1000
        print(df)
        ax = df.plot(x="time", y="temp",)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Temperature $(^\circ C)$")
        plt.title(f"200 inferences on {graph_name}")
        plt.savefig(f"{graph_name}.png")
