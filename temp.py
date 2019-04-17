import os
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from os import environ

def get_temp():
    with open('/sys/class/thermal/thermal_zone0/temp') as f:
        temp = int(f.readline())
        return temp

if __name__ == "__main__":
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/release'
    T = 5
    dt = 0.1
    i = 0
    x = []
    y = []
    x2 = []
    y2 = []
    for _ in range(int(T/dt)):
        x.append(i)
        y.append(get_temp())
        time.sleep(dt)
        i += dt
    st = i - dt
    p = subprocess.Popen(["./build/release/examples/graph_alexnet", "--target=NEON", "--therads=4"], env=env)
    while p.poll() == None:
        x.append(i)
        y.append(get_temp())
        time.sleep(dt)
        i += dt
        x2.append(x[-1])
        y2.append(y[-1])
    et = i - dt
    for _ in range(int(T/dt)):
        x.append(i)
        y.append(get_temp())
        time.sleep(dt)
        i += dt
    x3 = np.array(x2)
    y3 = np.array(y2)
    x3-=x3[0]
    y3-=y3[0]
    func = lambda t,q0,c,v,r:q0+c*v*(1-np.exp(- t/(r*c)))
    popt, _ = curve_fit(func,x3,y3, bounds=(0,10000))
    q0=popt[0]
    c=popt[1]
    v=popt[2]
    r=popt[3] 
    plt.plot(x, y)
    plt.plot(x3+x2[0], y2[0]+func(x3, *popt), 'r-', label='fit: Q0=%5.3f, C=%5.3f, V=%5.3f, R=%5.3f' % tuple(popt))
    plt.axvline(st, linestyle='--', color='y')
    plt.axvline(et, linestyle='--', color='y')
    plt.axhline(65000, linestyle='--', color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Execution')
    plt.legend(loc='best')
    plt.show()
    plt.savefig('temp_plots/graph_alexnet.png')
    
