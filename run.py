import os
import time
import multiprocessing
import subprocess
import random
import argparse
import time
from multiprocessing import Pool, Value
from threading import Thread, Lock
from os import environ

counter = None

def init(pcounter):
    global counter
    counter = pcounter

def run_graph(pargs):
    while True:
        with counter.get_lock():
            curr_idx = counter.value
            counter.value +=1

        if curr_idx < len(images):
            print(multiprocessing.current_process().name, curr_idx)
            #pargs.append("--image=" + images[curr_idx])
        
            subprocess.run(pargs, env=env)
            #time.sleep(random.randint(1, 2))
        else:
            return

cpu_args = ["./build/examples/graph_alexnet", "--target=NEON", "--threads=4"]
gpu_args = ["./build/examples/graph_alexnet", "--target=CL"] 
args = []
images = [""] * 100

if __name__ == "__main__":
    global env
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = './build/'
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", help="CPU", dest='cpu', action='store_true')
    ap.add_argument("--gpu", help="GPU", dest='gpu', action='store_true')
    cli_args = vars(ap.parse_args())
    if cli_args['cpu']:
        args.append(cpu_args)
    if cli_args['gpu']:
        args.append(gpu_args)
    if not cli_args['cpu'] and not cli_args['gpu']:
        ap.error('No target given, add --cpu or --gpu')
    counter = Value('i', 0)
    start = time.time()
    pool = Pool(initializer=init, initargs=(counter,), processes=len(args))
    pool.map(run_graph, args)
    end = time.time()
    print('Time taken: ', str((end - start) / 100.0), 'sec')
