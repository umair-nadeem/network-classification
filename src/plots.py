import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.outliers_influence import summary_table

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

import tensorflow as tf
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.initializers import glorot_normal
from keras.optimizers import SGD, rmsprop
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

from networks.search_batch_epoch import search_batch_epoch_grid
from networks.search_optimizer import search_optimizer_grid
from networks.search_rate_momentum import search_rate_momentum_grid
from networks.search_initialization import search_initialization_grid
from networks.search_activation import search_activation_grid
from networks.search_dropout import search_dropout_grid

from series.series_utilities import *
from series.series_methods import *
from series.trees_methods import *

from networks.netmodel import neural_prediction
from networks.neural_utils import split_into_chunks, mean_absolute_percentage_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/prediction_csv')

DATASET = 'short.csv'

df = pd.read_csv(DATASET, names=['time', 'len'], dtype=float, parse_dates=['time'])
df = df.set_index('time')
ts = df.squeeze()

ratio = 0.667
lag_depth = 24
horizon = 6
scale = ['1GB', 1, 'Hour']


def plot_predictions():
    df2 = pd.read_csv('/home/umair/PycharmProjects/thesis/files/'
                      'results/predictions/short_6_arima.csv', names=['time', 'len'], dtype=float, parse_dates=['time'])
    df2 = df2.set_index('time')
    short_6_arima = df2.squeeze()

    df2 = pd.read_csv('/home/umair/PycharmProjects/thesis/files/'
                      'results/predictions/short_6_smoothing.csv', names=['time', 'len'], dtype=float,
                      parse_dates=['time'])
    df2 = df2.set_index('time')
    short_6_smoothing = df2.squeeze()

    df2 = pd.read_csv('/home/umair/PycharmProjects/thesis/files/'
                      'results/predictions/short_6_forest.csv', names=['time', 'len'], dtype=float,
                      parse_dates=['time'])
    df2 = df2.set_index('time')
    short_6_forest = df2.squeeze()

    df2 = pd.read_csv('/home/umair/PycharmProjects/thesis/files/'
                      'results/predictions/short_6_neural.csv', names=['time', 'len'], dtype=float,
                      parse_dates=['time'])
    df2 = df2.set_index('time')
    short_6_neural = df2.squeeze()

    df2 = pd.read_csv('/home/umair/PycharmProjects/thesis/files/'
                      'results/predictions/short_6_gbm.csv', names=['time', 'len'], dtype=float, parse_dates=['time'])
    df2 = df2.set_index('time')
    short_6_gbm = df2.squeeze()

    train_size = int(len(ts) * ratio)
    train_ts = ts[np.arange(train_size)]
    test_ts = ts[train_size:]

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=1 - ratio)

    title = 'ST Series %d-%s Horizon Forecasting' \
            % (horizon * scale[1], scale[2])

    ylabel = 'Traffic Volume x %s' % scale[0]

    f = plt.figure()

    plt.subplot(511)
    plt.plot(pd.Series(y[-len(short_6_smoothing):], index=short_6_smoothing.index))
    plt.plot(short_6_arima, color='red', label='ARIMA')
    plt.title(title, fontsize=30)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.subplot(512)
    plt.plot(pd.Series(y[-len(short_6_smoothing):], index=short_6_smoothing.index))
    plt.plot(short_6_smoothing, color='red', label='Exponential Smoothing')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.subplot(513)
    plt.plot(pd.Series(y[-len(short_6_forest):], index=short_6_forest.index))
    plt.plot(short_6_forest, color='red', label='Random Forests')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    plt.ylabel(ylabel, fontsize=20)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.subplot(514)
    plt.plot(pd.Series(y[-len(short_6_neural):], index=short_6_neural.index))
    plt.plot(short_6_neural, color='red', label='Neural Network')

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.subplot(515)
    plt.plot(pd.Series(y[-len(short_6_gbm):], index=short_6_gbm.index))
    plt.plot(short_6_gbm, color='red', label='Gradient Boosted Trees')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)

    plt.xlabel('Time', fontsize=20)

    plt.subplots_adjust(hspace=0.1)
    f.show()


arima_real = [21.00, 21.99, 24.5, 25.95]
smoothing_real = [18.19, 19.90, 21.67, 21.18]
gbm_real = [17.62, 20.50, 21.83, 23.40]
forest_real = [19.31, 22.90, 23.95, 23.8]
nn_real = [17.68, 18.98, 19.19, 20.61]

horizon_real = [30, 60, 90, 120]

arima_short = [23.06, 23.46, 25.14, 26.52]
smoothing_short = [20.59, 23.42, 23.92, 26.04]
gbm_short = [21.57, 22.57, 24.47, 26.76]
forest_short = [22.20, 22.67, 26.07, 24.88]
nn_short = [18.38, 21.06, 22.28, 22.93]

horizon_short = [6, 12, 18, 24]

nn_mid = [22.97, 23.61, 25.89, 26.83]
arima_mid = [24.67, 26.62, 31.94, 37.25]
gbm_mid = [25.01, 27.70, 30.52, 35.42]
forest_mid = [23.75, 28.00, 32.73, 35.28]
smoothing_mid = [27.70, 26.21, 33.58, 34.36]

horizon_mid = [90, 180, 270, 365]


def plot_results():
    f = plt.figure()
    plt.plot(horizon_real, nn_real, marker='o', markersize=20, color='purple', label='Neural Network')
    plt.plot(horizon_real, arima_real, marker='s', markersize=20, color='brown', label='ARIMA')
    plt.plot(horizon_real, gbm_real, marker='*', markersize=20, color='green',
             label='Gradient Boosted Trees')
    plt.plot(horizon_real, forest_real, marker='p', markersize=20, color='blue', label='Random Forests')
    plt.plot(horizon_real, smoothing_real, marker='h', markersize=20, color='red',
             label='Exponential Smoothing')

    plt.title('RT Series Forecasting Mean Absolute Percentage Error (MAPE %)', fontsize=30)
    plt.ylabel('MAPE', fontsize=25)
    plt.xlabel('Horizon (Minutes)', fontsize=25)
    plt.xticks(horizon_real, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()

    g = plt.figure()
    plt.plot(horizon_short, nn_short, marker='o', markersize=20, color='purple', label='Neural Network')
    plt.plot(horizon_short, arima_short, marker='s', markersize=20, color='brown', label='ARIMA')
    plt.plot(horizon_short, gbm_short, marker='*', markersize=20, color='green',
             label='Gradient Boosted Trees')
    plt.plot(horizon_short, forest_short, marker='p', markersize=20, color='blue',
             label='Random Forests')
    plt.plot(horizon_short, smoothing_short, marker='h', markersize=20, color='red',
             label='Exponential Smoothing')

    plt.title('ST Series Forecasting Mean Absolute Percentage Error (MAPE %)', fontsize=30)
    plt.ylabel('MAPE', fontsize=25)
    plt.xlabel('Horizon (Hours)', fontsize=25)
    plt.xticks(horizon_short, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    g.show()

    h = plt.figure()
    plt.plot(horizon_mid, nn_mid, marker='o', markersize=20, color='purple', label='Neural Network')
    plt.plot(horizon_mid, arima_mid, marker='s', markersize=20, color='brown', label='ARIMA')
    plt.plot(horizon_mid, gbm_mid, marker='*', markersize=20, color='green',
             label='Gradient Boosted Trees')
    plt.plot(horizon_mid, forest_mid, marker='p', markersize=20, color='blue', label='Random Forests')
    plt.plot(horizon_mid, smoothing_mid, marker='h', markersize=20, color='red',
             label='Exponential Smoothing')

    plt.title('MT Series Forecasting Mean Absolute Percentage Error (MAPE %)', fontsize=30)
    plt.ylabel('MAPE', fontsize=25)
    plt.xlabel('Horizon (Days)', fontsize=25)
    plt.xticks(horizon_mid, fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    h.show()


def box_plots():
    labels = ['ARIMA', 'Exponential Smoothing', 'GBDT', 'Random Forests', 'Neural Network']
    boxprops = dict(linestyle='-', linewidth=4, color='brown')
    medianprops = dict(linestyle='--', linewidth=3, color='purple')

    f = plt.figure()
    plt.boxplot([arima_real, smoothing_real, gbm_real, forest_real, nn_real],
                labels=labels, boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of RT Series MAPE', fontsize=30)
    plt.ylabel('MAPE', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)
    plt.grid(True)
    f.show()

    g = plt.figure()
    plt.boxplot([arima_short, smoothing_short, gbm_short, forest_short, nn_short],
                labels=labels, boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of ST Series MAPE', fontsize=30)
    plt.ylabel('MAPE', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)
    plt.grid(True)
    g.show()

    h = plt.figure()
    plt.boxplot([arima_mid, smoothing_mid, gbm_mid, forest_mid, nn_mid],
                labels=labels, boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of MT Series MAPE', fontsize=30)
    plt.ylabel('MAPE', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)
    plt.grid(True)
    h.show()


def activ_plots():
    x = np.arange(-10, 10.01, 0.01)
    x1 = x[:1001]
    x2 = x[1001:]

    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    f = plt.figure()
    plt.plot(x, sigmoid, linewidth=3, color='blue', label='Logistic sigmoid')
    plt.plot(x, tanh, linewidth=3, color='brown', label='tanh')
    plt.title('Sigmoid and tanh Activation Function', fontsize=30)
    plt.xlabel('x', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([-10, 10])
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()

    relu = np.append(np.zeros(1001), x2)

    g = plt.figure()
    plt.plot(x, relu, linewidth=3, color='purple')
    plt.title('ReLU', fontsize=30)
    plt.xlabel('z', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([-3, 10])
    plt.xlim([-10, 10])
    plt.grid(True)
    g.show()

    lrelu = np.append(0.05 * x1, x2)
    elu = np.append(0.05 * (np.exp(x1) - 1), x2)

    h = plt.figure()
    plt.plot(x, relu, linewidth=3, color='purple', label='ReLU')
    plt.plot(x, lrelu, linewidth=3, color='green', label='LReLU')
    plt.plot(x, elu, linewidth=3, color='blue', label='ELU')
    plt.title('ReLU, LReLU and ELU Activation Function', fontsize=30)
    plt.xlabel('z', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([-3, 10])
    plt.xlim([-10, 10])
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    h.show()
