import os
import time
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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

temp_func = lambda t, q0, qi, tau: q0 + qi * (1 - np.exp(- t / tau))


def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

def plot_temp(xs, ys):
    fig = plt.figure()
    plt.plot(xs, ys)
    plt.axhline(TL, linestyle='--', color='r')
    mx = int(np.max(xs))
    for i in np.arange(0, mx, sdt):
        plt.axvline(i, linestyle='--', color='y')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution ' + graph)
    plt.show()
       
def run_sched():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    #time.sleep(T)
    cmds = list(targets[target](graph) for target in targets)
    print(cmds)
    ps = list(subprocess.Popen(cmd, env=env) for cmd in cmds)
    completed = False
    xs = []
    ys = []
    t=0
    while not completed:
        for _ in range(int(sdt/dt)):
            if None in (p.poll() for p in ps):
                xs.append(t)
                ys.append(get_temp())
                time.sleep(dt)
                t+=dt
            else:
                completed = True
                break
    xs=np.array(xs)
    ys=np.array(ys)
    plot_temp(xs, ys)
            
if __name__ == "__main__":
    run_sched()
