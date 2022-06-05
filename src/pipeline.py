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
from networks.search_grid import grid_search
from networks.neural_utils import split_into_chunks, mean_absolute_percentage_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/prediction_csv')

"""The three datasets are as follows:
1.  Middle-term dataset (mid.csv) from Oct 2006 to Dec 2017
    is 11.25 years long.
    Frequency:      5-day
    Scale:          100GB
    Horizon (h):    3-75 (73)
    Cycle:          73
    order:          Non-seasonal ARIMA parameters: [3, 1, 0], [0, 1, 2], [3, 1, 2]
                    other possibilites: [3, 1, 0], [0, 1, 5], [3, 1, 5]
    net_params:     [80, 20, 'adamax', 'relu', 'lecun_uniform', 0.0, 0.8, [10, 1]]

2.  Short-term dataset (short.csv) from 06 June to 15 July 2007 is
    40 days long.
    Freqency:       1-hour
    Scale:          1GB
    Horizon (h):    1-24 (24)
    Cycle:          168
    order:          [1, 1, 0] (Sarimax. Not possible [0, 1, 1], [1, 1, 1], [0, 1, 2], [1, 1, 2])
    net_params:     [80, 10, 'Nadam', 'relu', 'glorot_normal', 0.0, 1, [4, 1]]

3.  Real-time dataset (real.csv) from 06 June to 15 June 2007 is
    10 days long.
    Freqency:       5-Min
    Scale:          100MB
    Horizon (h):    1-24 (24)
    Cycle:          288
    order:          [5, 1, 0] (Sarimax. Not possible [0, 1, 2], [5, 1, 2])
    net_params:     [[80, 10, 'Adadelta', 'elu', 'lecun_normal', 0.0, 0.8, [4, 1]]
    """

DATASET = 'real.csv'

df = pd.read_csv(DATASET, names=['time', 'len'], dtype=float, parse_dates=['time'])
df = df.set_index('time')
ts = df.squeeze()

params = {'mid.csv': {'lag_depth': 73,
                      'horizon': 73,
                      'cycle': 73,
                      'order': [3, 1, 2],
                      'sorder': [0.61, 0.005, 0.0],
                      'scale': ['100GB', 5, 'Day'],
                      'net_params': [80, 10, 'Nadam', 'relu', 'lecun_uniform', 0.0, 0.8, [12, 1]],
                      'xgb_params': {'objective': 'mean_absolute_percentage_error',
                                     'metric': 'mean_absolute_percentage_error',
                                     'boosting_type': 'gbdt',
                                     'learning_rate': 0.2,
                                     'max_depth': 15,
                                     'num_leaves': 80,
                                     'verbose': -1,
                                     'min_data_in_leaf': 20,
                                     'seed': 47},
                      'forest_params': {'n_estimators': 40,
                                        'criterion': 'mae',
                                        'min_samples_leaf': 5,
                                        'max_depth': 15,
                                        'n_jobs': -1,
                                        'random_state': 47,
                                        'verbose': 0}},
          'short.csv': {'lag_depth': 168,
                        'horizon': 24,
                        'cycle': 168,
                        'order': [1, 1, 0],
                        'sorder': [0.2, 0.0, 0.32],
                        'scale': ['1GB', 1, 'Hour'],
                        'net_params': [80, 10, 'Nadam', 'relu', 'glorot_normal', 0.0, 1, [4, 1]],
                        'xgb_params': {'objective': 'mean_absolute_percentage_error',
                                       'metric': 'mean_absolute_percentage_error',
                                       'boosting_type': 'gbdt',
                                       'learning_rate': 0.15,
                                       'max_depth': 4,
                                       'num_leaves': 35,
                                       'min_data_in_leaf': 4,
                                       'verbose': -1,
                                       'seed': 47},
                        'forest_params': {'n_estimators': 30,
                                          'criterion': 'mae',
                                          'min_samples_leaf': 1,
                                          'max_depth': 14,
                                          'n_jobs': -1,
                                          'random_state': 47,
                                          'verbose': 0}},
          'real.csv': {'lag_depth': 288,
                       'horizon': 18,
                       'cycle': 288,
                       'order': [5, 1, 1],
                       'sorder': [0.2, 0.0, 0.39],
                       'scale': ['100MB', 5, 'Min'],
                       'net_params': [80, 10, 'AdaDelta', 'elu', 'lecun_normal', 0.0, 0.7, [4, 1]],
                       'xgb_params': {'objective': 'mean_absolute_percentage_error',
                                      'metric': 'mean_absolute_percentage_error',
                                      'boosting_type': 'gbdt',
                                      'learning_rate': 0.08,
                                      'max_depth': 3,
                                      'num_leaves': 40,
                                      'verbose': -1,
                                      'min_data_in_leaf': 4,
                                      'seed': 47},
                       'forest_params': {'n_estimators': 30,
                                         'criterion': 'mae',
                                         'min_samples_leaf': 1,
                                         'max_depth': 15,
                                         'n_jobs': -1,
                                         'random_state': 47,
                                         'verbose': 0
                                         }}}

ratio = 0.667

horizon = params[DATASET]['horizon']
cycle = params[DATASET]['cycle']
order = params[DATASET]['order']
lag_depth = params[DATASET]['lag_depth']
scale = params[DATASET]['scale']

alpha = params[DATASET]['sorder'][0]  # for preceding values
beta = params[DATASET]['sorder'][1]  # for trend slope
gamma = params[DATASET]['sorder'][2]

batch_size = params[DATASET]['net_params'][0]
epochs = params[DATASET]['net_params'][1]
optimizer = params[DATASET]['net_params'][2]
activation = params[DATASET]['net_params'][3]
initializer = params[DATASET]['net_params'][4]
dropout_rate = params[DATASET]['net_params'][5]
weight_constraint = params[DATASET]['net_params'][6]
neurons = params[DATASET]['net_params'][7]

xgb_params = params[DATASET]['xgb_params']
forest_params = params[DATASET]['forest_params']

loss = 'mean_absolute_percentage_error'

batch_values = [80]
epoch_values = [10, 20]

model, mape, pred = forest(ts, forest_params, ratio, horizon, lag_depth, scale)

"""
model_name = 'neural'
pred_name = DATASET[:-4] + '_' + str(horizon) + '_' + model_name + DATASET[-4:]
pred.to_csv(pred_name, header=None)


model, mape, pred = nstep_arima(ts, ratio, horizon, order, scale)
model, mape, pred = sarimax(ts, ratio, horizon, order, cycle, scale)
model, mape, pred = nstep_smoothing(ts, ratio, horizon, cycle, alpha, beta, gamma, scale)
model, mape, pred = xgb(ts, xgb_params, ratio, horizon, lag_depth, scale)
model, mape, pred = forest(ts, forest_params, ratio, horizon, lag_depth, scale)

history, mape, pred = neural_prediction(batch_size, epochs, ts, ratio, horizon,
                                        lag_depth, neurons, activation, initializer,
                                        loss, optimizer, dropout_rate,
                                        weight_constraint, scale)
                                       
grid_search(batch_size, epochs, ts, ratio, 
            lag_depth, neurons, initializer,
            loss)

grid = search_batch_epoch_grid(batch_values, epoch_values, ts, ratio,
                               lag_depth, neurons, activation, initializer,
                               loss, optimizer)

grid = search_optimizer_grid(batch_size, epochs, ts, ratio,
                             lag_depth, neurons, activation,
                             initializer, loss)

grid = search_activation_grid(batch_size, epochs, ts, ratio,
                              lag_depth, neurons, initializer,
                              loss, optimizer)

grid = search_initialization_grid(batch_size, epochs, ts, ratio,
                                  lag_depth, neurons, activation,
                                  loss, optimizer)

grid = search_rate_momentum_grid(batch_size, epochs, ts, ratio,
                                 lag_depth, neurons)

grid = search_dropout_grid(batch_size, epochs, ts, ratio,
                           lag_depth, neurons)

tw = history.model.trainable_weights
ot = history.model.output
gradients = k.gradients(ot, tw)

trainingExample = np.random.random((neurons[-1], lag_depth))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evg = sess.run(gradients, feed_dict={history.model.input: trainingExample})

"""
