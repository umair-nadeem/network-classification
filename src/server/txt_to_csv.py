import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/home/umair/PycharmProjects/thesis/files/WAIKATO_8/csv')

list_names = os.listdir()
list_names.sort()

series = pd.Series()

for i in range(0, len(list_names)):
    f_name = list_names[i]
    print(f_name)

    df = pd.read_csv(f_name, names=['time', 'len'], dtype=float, parse_dates=['time'])
    df = df.set_index('time')
    ts = df.squeeze()

    series = series.append(ts)