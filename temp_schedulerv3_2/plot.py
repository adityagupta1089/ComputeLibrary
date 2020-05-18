import matplotlib.pyplot as plt
import os
import re


def plot_temp(TL, dt):
    plt.figure()
    plt.plot(x, y)
    plt.axhline(80000, linestyle='--', color='r')
    if TL != 80000 and TL <= 100000:
        plt.axhline(TL, linestyle='--', color='g')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution scheduler v3.2 (TL='+str(TL)+',dt='+str(dt)+'')
    plt.savefig('resnet50_TL'+str(TL)+'_dt'+str(dt)+'.png')


for file in os.listdir():
    if re.match(r"resnet50_TL\d+_dt\d+.log", file):
        with open(file, 'r') as f:
            content = f.readlines()
        tl, dt = re.findall(r"resnet50_TL(\d+)_dt(\d+).log", file)[0]
        tl = int(tl)
        dt = int(dt)
        print(tl, dt)

        x = []
        y = []
        for line in content[1:]:
            xx, yy = line.split(',')
            x.append(float(xx))
            y.append(float(yy))

        plot_temp(tl, dt)
