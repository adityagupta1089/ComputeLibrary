import os
import time
import subprocess
import matplotlib.pyplot as plt

from os import environ

def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

if __name__ == "__main__":
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    i = 0
    x = []
    y = []
    for _ in range(5):
        x.append(i)
        y.append(get_temp())
        time.sleep(1)
        i += 1
    st = i - 1
    p = subprocess.Popen(["./build/release/examples/graph_alexnet", "--target=NEON", "--therads=4"], env=env)
    while p.poll() == None:
        x.append(i)
        y.append(get_temp())
        time.sleep(1)
        i += 1
    et = i - 1
    for _ in range(5):
        x.append(i)
        y.append(get_temp())
        time.sleep(1)
        i += 1
    plt.plot(x, y)
    plt.axvline(st, linestyle='--', color='y')
    plt.axvline(et, linestyle='--', color='y')
    plt.axhline(65000, linestyle='--', color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution')
    plt.savefig('temp_plots/graph_alexnet.png')
    
