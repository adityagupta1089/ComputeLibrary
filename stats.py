import glob
import pandas as pd
import re
from collections import defaultdict

temp_csvs = glob.glob("temp_schedulerv*/*_TL*_dt*.csv")
time_logs = glob.glob("temp_scheduler_all/*_TL*_dt*_run_sched*.log")


def dd():
    return defaultdict(dd)


stats = defaultdict(dd)

for temp_csv in temp_csvs:
    version, graph, TL = re.findall(
        r"temp_schedulerv([\d_]+)/(\w+)_TL(\d+)_dt\d+.csv", temp_csv
    )[0]
    version = version.replace("_", ".")
    df = pd.read_csv(temp_csv)
    average_temp = df[" temp"].mean()
    crosses_threshold = len(df[df[" temp"] > int(TL)]) / len(df[" temp"]) * 100
    stats[graph][version][TL]["average_temp"] = average_temp
    stats[graph][version][TL]["crosses_threshold"] = crosses_threshold

for time_log in time_logs:
    file_name = time_log.split("/")[1]
    graph, TL, version = re.findall(
        r"(\w+)_TL(\d+)_dt\d+_run_sched([\d\.]+).log", file_name
    )[0]
    with open(time_log) as f:
        content = f.read()
    time = re.findall(r"([\d\.]+) per inference", content)[0]
    stats[graph][version][TL]["time_taken"] = time

for graph in stats:
    print(f"Graph = {graph}")
    for version in sorted(stats[graph]):
        if version not in ("3", "3.1", "4"):
            continue
        print(f"    Version = {version}")
        for TL in sorted(stats[graph][version]):
            if TL not in ("80000", "999999"):
                continue
            print(f"        TL = {TL}")
            data = stats[graph][version][TL]
            print(f"            Average temp = {data['average_temp']:.2f}")
            print(
                f"            Crosses threshold = {data['crosses_threshold']:.2f}"
            )
            print(f"            Time taken = {data['time_taken']}")
