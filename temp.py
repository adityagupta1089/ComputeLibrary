import os
import time
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from os import environ

pp = "./build/release/examples/"

graphs = [
    "graph_alexnet",
    "graph_mobilenet",
    "graph_resnet50",
]

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

temp_func = lambda t, q0, qi, tau: q0 + qi * (1 - np.exp(- t / tau))


def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

def plot_temp(graph, target, x, y, popt, i):
    fig = plt.figure()
    plt.plot(x, y)
    plt.plot(x, temp_func(x, *popt), 
        'r-', 
        label='fit: Q0=%5.3f, Qinf=%5.3f, Tau=%5.3f' % tuple(popt))
    plt.axhline(65000, linestyle='--', color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution ' + graph + " " + target)
    plt.legend(loc='best')
    plt.savefig('temp_plots/' + graph + "_" + target + "_" + str(i) + ".png")  
    plt.close(fig)

def get_params(graph, target, x, y, i):
    try:
        popt, _ = curve_fit(temp_func, x, y, bounds=(0, np.inf))
        print('> Q0=%5.3f, Qinf=%5.3f, Tau=%5.3f' % tuple(popt))
        plot_temp(graph, target, x, y, popt, i)
        return popt
    except:
        print("> Bad Graph")
        return []
       
def find_params(graph, target, cmd, i):
    x = []
    y = []
    t = 0
    time.sleep(T)
    p = subprocess.Popen(cmd, env=env)
    while p.poll() == None:
        x.append(t)
        y.append(get_temp())
        time.sleep(dt)
        t += dt
    x = np.array(x)
    y = np.array(y)
    return get_params(graph, target, x, y, i)
            
if __name__ == "__main__":
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    RCs = {}
    for graph in graphs:
        print(graph)
        for target in targets:
            print(target)
            for i in range(5):
                RCs[(graph, target)] = find_params(graph, target, targets[target](graph), i)
