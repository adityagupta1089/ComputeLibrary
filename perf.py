import os
import re
import subprocess
import matplotlib
import json
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata

pp = "./build/release/examples/"

graphs = [
    "graph_alexnet_2",
    "graph_mobilenet_2",
    "graph_googlenet_2",
    "graph_squeezenet_2",
    "graph_resnet50_2"
]

targets = [
    "--cpu",
    "--gpu",
    "--cpu --gpu"
]

def persist_to_file(file_name):
    def decorator(func):
        try:
            cache = json.load(open(file_name, 'r'))
        except (IOError, ValueError):
            cache = {}
        def new_func(g,t,n,i):
            k1 = g + " " + t.replace('-', '')
            k2 = str(n) + " " + str(i)
            if k1 not in cache:
                cache[k1] = {}
            if k2 not in cache[k1]:
                cache[k1][k2] = func(g,t,n,i)
                json.dump(cache, open(file_name, 'w'))
            return cache[k1][k2]
        return new_func
    return decorator

@persist_to_file('perf_cache.dat')
def run(graph, target, n, i):
    global env
    cmd = pp + graph + " " + target + " --n=" + str(n) + " --i=" + str(i)
    out = subprocess.check_output(cmd.split(), env=env).decode('utf-8') 
    cpu_images = 0
    gpu_images = 0
    if "cpu" in target:
        cpu_images = int(re.findall('Completed CPU: (\d+)', out)[0])
    if "gpu" in target:
        gpu_images = int(re.findall('Completed GPU: (\d+)', out)[0])
    t_image = float(re.findall('(\d+.\d+) per image', out)[0])
    return cpu_images, gpu_images, t_image

def plot_fig(k, x, y, z):
    graph, target = k
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1 = np.linspace(min(x), max(x), len(set(x)))
    y1 = np.linspace(min(y), max(y), len(set(y)))
    x1, y1 = np.meshgrid(x1, y1)
    z1 = griddata((x, y), z, (x1, y1), method='cubic')
    surf = ax.plot_surface(x1, y1, z1, cmap='jet')
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('# Images')
    ax.set_ylabel('# Inferences/Images')
    ax.set_zlabel('Time (sec)')
    ax.set_title('Total Time Taken')
    plt.savefig('perf_plots/'+graph+"_"+target+"_2.png")

if __name__ == "__main__":
    global env
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    time = {}
    for graph in graphs[:1]:
        for target in targets[:1]:
            _target = target.replace('-', '')
            time[(graph, _target)] = []
            for n in range(10, 81, 10):
                for i in range(10, 101, 10):
                    if n == 80 and i == 60:
                        break
                    ci, gi, tn = run(graph, target, n, i)
                    print(n, i, tn * (ci + gi))
                    time[(graph, _target)].append((n, i, tn * (ci + gi), ci, gi))

    for k in time:
        xyz = list(zip(*time[k]))[:3]
        x, y, z = np.array(xyz[0]), np.array(xyz[1]), np.array(xyz[2])
        plot_fig(k, x, y, z)
        def func(xy, a, b, c, d, e, f):
            [x, y] = xy.T
            return a*x + b*y + c*x*x+d*y*y+e*x*y+f
        popt, pcov = curve_fit(func, np.array(list(zip(x, y))), z)
        print("Std. Error:", np.sqrt(np.diag(pcov)))
        print("ax+by+cx^2+dy^2+exy+f")
        print("a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f, f=%5.3f" % tuple(popt))

