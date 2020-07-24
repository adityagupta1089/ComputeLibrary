import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

matplotlib.style.use("bmh")

df = pd.read_csv("resnet50_TL80000_dt1000_motivation.csv")
df["temp"] /= 1000
sns.lineplot(data=df, x="time", y="temp")
plt.axhline(80, linestyle="--", color="r")
plt.xlabel("Time (sec)")
plt.ylabel("Temperature $(^\circ C)$")
plt.title("Execution scheduler resnet50, AMTRI, ${{\\rm TL}} = 80^\circ$")
# plt.show()
print("Saving")
plt.savefig("resnet50_TL80000_dt1000_motivation.png")
