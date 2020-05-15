from os import environ
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from random import randint
import subprocess
from itertools import combinations
import matplotlib

matplotlib.use('Agg')


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
dt = 0.01


def temp_func(t, b, c): return b * (1 - np.exp(- t / c))


def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

def plot_temp(cmd_combi, x, y, popt, i, t0):
    fig = plt.figure()
    plt.plot(x, [t0+yy for yy in y])
    yp = [t0+temp_func(xx, *popt) for xx in x]
    plt.plot(x, yp, 
        'r-', 
        label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
    plt.axhline(80000, linestyle='--', color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    combi_name = "-".join([target for target, _ in cmd_combi])
    plt.title('Execution ' + graph + " " + combi_name)
    plt.legend(loc='best')
    name ='temp_profile_v4/' + graph + "_" + combi_name + "_" + str(i) + ".png"
    print(name)
    plt.savefig(name)  
    plt.close(fig)

if __name__ == "__main__":
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    RCs = {}
    cmd_combis = []
    cmds = list((target, targets[target](graph)) for target in targets)
    for i in range(1, len(cmds) + 1):
        cmd_combis += list(combinations(cmds, i))
    for cmd_combi in cmd_combis:
        print('> ', "-".join([target for target, _ in cmd_combi]))
        bs = 0
        cs = 0
        bp = 30000
        cp = 5
        for i in range(10):
            time.sleep(randint(0, T))
            t=0
            running = []
            for target_cmd in cmd_combi:
                running.append(subprocess.Popen(target_cmd[1], env=env))
            x=[]
            y = []
            while None in (p.poll() for p in running):
                x.append(t)
                y.append(get_temp())
                time.sleep(dt)
                t+=dt
            y.append(get_temp())
            x.append(t)
            t0 = y[0]
            mx = t0
            y = [yy-t0 for yy in y]
            popt, _ = curve_fit(temp_func, x, y, bounds=(0, np.inf), p0=(bp, cp))
            [b, c] = popt
            bp = b
            cp = c
            bs += b
            cs += c
            print('b=%5.3f, c=%5.3f' % (b, c))
            plot_temp(cmd_combi, x, y, popt, i, t0)
        bs /= 10
        cs /= 10
        print('Average: b=%5.3f, c=%5.3f' % (bs, cs))
