import os
import time
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from os import environ

pp = "./build/release/examples/"

graph = "graph_resnet50"

cp = ["--target=NEON", "--threads=4"]
gp = ["--target=CL"]
sm = ["taskset", "-c", "0-3"]
bg = ["taskset", "-c", "4-7"]

targets = {
    "cpu_big": lambda g: bg + [pp + g] + cp,
    "cpu_small": lambda g: sm + [pp + g] + cp,
    "gpu": lambda g: [pp + g] + gp
}

T = 5
dt = 0.1
sdt = 0.5
TL = 65000
N = 50

def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

def plot_temp(xs, ys, xs2, ys2):
    fig = plt.figure()
    # Plot Lines
    for x, y in zip(xs, ys):
        plt.plot(x, y, color='g')
    for x2, y2 in zip(xs2, ys2):
        plt.plot(x2, y2, color='b')
    # Plot Threshold
    plt.axhline(TL, linestyle='--', color='r')
    # Plot Labels
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution ' + graph)
    # Plot Legend
    custom_lines = [
        Line2D([0], [0], color='r', linestyle='--'),
        Line2D([0], [0], color='g'),
        Line2D([0], [0], color='b')]
    plt.legend(custom_lines, ['Threshold', 'Executing', 'Sleeping'])
    # Save Plot
    plt.savefig('tempsched_plots/sched.png')
       
def run_sched():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    cmds = list(targets[target](graph) for target in targets)
    xs = []
    ys = []
    xs2 = []
    ys2 = []
    t=0
    n=0
    # sleep to cool down
    time.sleep(T)
    while n < N:
        ps = list(subprocess.Popen(cmd, env=env) for cmd in cmds)
        completed = False
        while not completed:
            if get_temp() > TL:
                x2 = []
                y2 = []
                for _ in range(int(sdt/dt)):
                    x2.append(t)
                    y2.append(get_temp())
                    time.sleep(dt)
                    t+=dt
                x2.append(t)
                y2.append(get_temp())
                xs2.append(x2)
                ys2.append(y2)
            else:
                x = []
                y = []
                for _ in range(int(sdt/dt)):
                    if None in (p.poll() for p in ps):
                        x.append(t)
                        y.append(get_temp())
                        time.sleep(dt)
                        t+=dt
                    else:
                        completed = True
                        n += 3
                        print('Completed', n)
                        break
                x.append(t)
                y.append(get_temp())
                xs.append(x)
                ys.append(y)
    plot_temp(xs, ys, xs2, ys2)
            
if __name__ == "__main__":
    run_sched()
