import os
import time
import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse

from matplotlib.lines import Line2D
from os import environ
from itertools import combinations

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
TL = 80000
N = 50

def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

def profile_temp():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    cmds = list((target, targets[target](graph)) for target in targets)
    cmd_combis = []
    for i in range(1, len(cmds) + 1):
        cmd_combis += list(combinations(cmds, i))
    samples = dict()
    for _ in range(10):
        for cmd_combi in cmd_combis:
            time.sleep(T)
            running = []
            for target_cmd in cmd_combi:
                running.append(subprocess.Popen(target_cmd[1], env=env))
            y = []
            while None in (p.poll() for p in running):                
                y.append(get_temp())
                time.sleep(dt)
            y.append(get_temp())
            max_dt = max(y) - y[0]
            combi_targets = [x[0] for x in cmd_combi]
            key = '_'.join(combi_targets)
            if key not in samples:
                samples[key] = []
            samples[key].append(max_dt)
            print(key, max_dt)
            with open('tempschedv3_samples.json', 'w') as f:
                json.dump(samples, f)

def profile_time():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    samples = dict()
    for _ in range(10):
        for target in targets:
            time.sleep(T)
            start = time.time()
            subprocess.Popen(targets[target](graph), env=env)
            end = time.time()
            if target not in samples:
                samples[target] = []            
            samples[target].append(end - start)
            print(target, end-start)
            with open('tempschedv3_samples2.json', 'w') as f:
                json.dump(samples, f)

def plot_temp(xs, ys, xs2, ys2):
    fig = plt.figure()
    # Plot Lines
    for x, y in zip(xs, ys):
        plt.plot(x, y, color='r')
    for x2, y2 in zip(xs2, ys2):
        plt.plot(x2, y2, color='y')
    # Plot Threshold
    plt.axhline(TL, linestyle='--', color='r')
    # Plot Labels
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution ' + graph)
    # Plot Legend
    custom_lines = [
        Line2D([0], [0], color='r', linestyle='--'),
        Line2D([0], [0], color='r'),
        Line2D([0], [0], color='y')]
    plt.legend(custom_lines, ['Threshold', 'Executing', 'Sleeping'])
    # Show Plot
    plt.savefig('tempsched_plots/schedv3.png')
       
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
    ps = dict()
    with open('tempschedv3_samples.json') as f:
        temp_samples = json.load(f)
    with open('tempschedv3_samples2.json') as f:
        time_samples = json.load(f)
    for target in time_samples:
        time_samples[target] = sum(time_samples[target]) / len(time_samples[target])    
    for target_combi in temp_samples:
        temp_samples[target_combi] = sum(temp_samples[target_combi]) / len(temp_samples[target_combi])    
    min_dT = min(temp_samples.values())
    # sleep to cool down
    print('Sleeping 5 sec')
    time.sleep(T)
    while n < N:
        if get_temp() + min_dT > TL:
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
            running = dict()
            for target in targets:
                running[target] = False
            while get_temp() + min_dT < TL and n < N:
                dT = TL - get_temp() # temperature difference
                min_t = 99999999 # time taken
                min_combi = "none"
                for target_combi in temp_samples:
                    valid = True
                    # check temperature difference
                    if temp_samples[target_combi] < dT:
                        # measure sum of execution times of individual targets
                        tmes = 0
                        for target in targets:
                            # check if combination also includes running targets
                            if running[target] and target not in target_combi:
                                valid=False
                                break
                            # we can add running target's sample vaue but it will be constant across comparison
                            if target in target_combi:
                                tmes += time_samples[target]
                        if not valid:
                            continue
                        # take maximum performance
                        if tmes < min_t:
                            min_t = tmes
                            min_combi = target_combi      
                for target, cmd in zip(targets, cmds):
                    if not target in ps:
                        ps[target] = None
                    # if empty process or not running
                    if ps[target] is None or ps[target].poll() is not None:
                        if running[target] and ps[target] is not None:
                            running[target] = False
                            n+=1
                            print('Completed', target, n)
                            if n > N:
                                break
                        elif target in min_combi:
                            running[target] = True
                            ps[target] = subprocess.Popen(cmd, env=env) 
                x.append(t)
                y.append(get_temp())
                time.sleep(dt)
                t+=dt
            x.append(t)
            y.append(get_temp())
            xs.append(x)
            ys.append(y)
    plot_temp(xs, ys, xs2, ys2)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-temp', default=False, help="Profile Temperature")
    parser.add_argument('--profile-time', default=False, help="Profile Time")
    parser.add_argument('--run-sched', default=False, help="Run Scheduler")
    args = parser.parse_args()
    if args.profile_temp:
        profile_temp()
    if args.profile_time:
        profile_time()
    if args.run_sched:
        run_sched()
