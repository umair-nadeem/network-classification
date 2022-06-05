import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib

import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.outliers_influence import summary_table

os.chdir('/home/umair/project/py_scripts')

DATASET = 'mid.csv'

df = pd.read_csv(DATASET, names=['time', 'len'], dtype=float, parse_dates=['time'])
df = df.set_index('time')
ts = df.squeeze()

params = {'mid.csv': {'cycle': 168,
                      'order': [1, 1, 0]}}

ratio = 0.667

cycle = params[DATASET]['cycle']
order = params[DATASET]['order']


def get_regression_score(y_true, y_pred):
    """This method receives numpy arrays y_true and y_pred
    and calculates three performance metrics i.e. mean
    squared error, mean absolute error and mean absolute
    percentage error."""
    y_true[np.where(y_true == 0.0)] = 0.001
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.

    print('\nMean Absolute Percentange Error (MAPE): %.2f\n' % mape)

    return mape


def sarimax(ts, ratio, h, order, winlen):
    train_size = int(len(ts) * ratio)
    train_ts = ts[np.arange(train_size)]
    test_ts = ts[train_size:]
    history = train_ts
    y_pred = pd.Series()

    for t in range(0, len(test_ts), h):
        print('\nBuilding the model at %s ' % datetime.now())
        model = sm.tsa.statespace.SARIMAX(history, order=order, freq=pd.infer_freq(ts.index),
                                          seasonal_order=(order[0], order[1], order[2], winlen),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

        model_fit = model.fit(disp=False)

        y_test = test_ts[t:t + h]

        pred_results = model_fit.get_prediction(start=y_test.index[0],
                                                end=y_test.index[-1], dynamic=True)

        y_temp_pred = pred_results.predicted_mean

        y_pred = y_pred.append(y_temp_pred)

        history = history.append(y_test)

        print("Horizon %d is %.2f percent complete... at: %s" %
              (h, 100 * min((t + h), len(test_ts)) / len(test_ts), datetime.now()))

    file_name = 'y_pred_horizon_' + str(h) + '.csv'
    y_pred.to_csv(file_name, index=None)

    mape = get_regression_score(test_ts.values, y_pred.values)

    return mape


horizons = np.arange(1, 25, 1)
error = np.asarray([])

for hn in horizons:
    print('Starting predictions for horizon %d at %s: ' % (hn, datetime.now()))
    er = sarimax(ts, ratio, hn, order, cycle)
    error = np.append(error, er)

mean_error = pd.Series(error, index=horizons)
mean_error.to_csv('mape_mid_order110.csv', header=None)

print('Saved to file!')
