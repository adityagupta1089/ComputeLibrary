from os import environ
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import subprocess
from itertools import combinations
import matplotlib


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


if __name__ == "__main__":
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    RCs = {}
    cmd_combis = []
    cmds = list((target, targets[target](graph)) for target in targets)
    for i in range(1, len(cmds) + 1):
        cmd_combis += list(combinations(cmds, i))
    for cmd_combi in cmd_combis:
        print('> ', cmd_combi)
        bs = 0
        cs = 0
        for i in range(10):
            time.sleep(T)
            running = []
            for target_cmd in cmd_combi:
                running.append(subprocess.Popen(target_cmd[1], env=env))
            y = []
            while None in (p.poll() for p in running):
                y.append(get_temp())
                time.sleep(dt)
            y.append(get_temp())
            y = [yy-y[0] for yy in y]
            popt, _ = curve_fit(temp_func, x, y, bounds=(0, np.inf))
            [b, c] = popt
            bs += b
            cs += c
            print('b=%5.3f, c=%5.3f' % (b, c))
        bs /= 10
        cs /= 10
        print('Average: b=%5.3f, c=%5.3f' % (bs, cs))
