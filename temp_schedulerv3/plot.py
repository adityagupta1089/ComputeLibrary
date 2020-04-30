import matplotlib.pyplot as plt
import os
import re


def plot_temp(TL):
    plt.figure()
    plt.plot(x, y)
    plt.axhline(80000, linestyle='--', color='r')
    if TL != 80000 and TL <= 100000:
        plt.axhline(TL, linestyle='--', color='g')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title(f'Execution scheduler v3 (TL={TL})')
    plt.savefig(f'resnet50_TL{TL}.png')


for file in os.listdir():
    if re.match(r"resnet50_TL\d+.csv", file):
        with open(file, 'r') as f:
            content = f.readlines()
        tl = int(re.findall(r"resnet50_TL(\d+).csv", file)[0])
        print(tl)

        x = []
        y = []
        for line in content[1:]:
            xx, yy = line.split(',')
            x.append(float(xx))
            y.append(float(yy))

        plot_temp(tl)
