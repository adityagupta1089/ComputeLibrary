import pandas as pd
import re
import os

for file in os.listdir():
    if re.match(r"resnet50_TL\d+.csv", file):
        df = pd.read_csv(file)
        tl = re.findall(r"resnet50_TL(\d+).csv", file)[0]
        print(f"tl = {tl} =====")
        print(df.mean())
        print(f'T>TL = { len(df[df[" temp"] > int(tl)]) / len(df[" temp"])}')
