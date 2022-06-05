import numpy as np
import pandas as pd
import pickle
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .series_utilities import *

plt.style.use('seaborn')


def nstep_arima(ts, ratio, horizon, order, scale):
    if ratio > 1 or ratio <= 0:
        print("\nWrong value of train ratio (train data/total data). Exiting!\n")
        return

    train_size = int(len(ts) * ratio)
    train_ts = ts[np.arange(train_size)]
    test_ts = ts[train_size:]
    history = train_ts
    y_pred = pd.Series()
    model_fit = None

    for t in range(0, len(test_ts), horizon):
        model = ARIMA(history, order=order, freq=pd.infer_freq(ts.index))
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast(np.min([len(test_ts) - t, horizon]))[0]

        history = history.append(test_ts[t:t + horizon])
        y_pred = y_pred.append(pd.Series(yhat,
                                         index=test_ts.index[t:t + horizon]))

        print("%.2f percent complete..." % (100 * min((t + horizon), len(test_ts)) / len(test_ts)))

    mape = get_regression_score(test_ts.values, y_pred.values)

    title = '%d-%s Horizon Forecasting with ARIMA (MAPE: %.2f%%)' \
            % (horizon * scale[1], scale[2], mape)
    ylabel = 'Traffic Volume x %s' % scale[0]

    plot_predictions(test_ts, y_pred, title, ylabel)
    plot_error_density(model_fit)

    return model_fit, mape, pd.Series(y_pred, index=test_ts.index)


def sarimax(ts, ratio, h, order, winlen, scale):
    train_size = int(len(ts) * ratio)
    train_ts = ts[np.arange(train_size)]
    test_ts = ts[train_size:]
    history = train_ts
    y_pred = pd.Series()
    model_fit = None

    for t in range(0, len(test_ts), h):
        print(datetime.now())
        model = sm.tsa.statespace.SARIMAX(history, order=order, freq=pd.infer_freq(ts.index),
                                          seasonal_order=(order[0], order[1], order[2], winlen),
                                          enforce_stationarity=False,
                                          enforce_invertibility=False)

        model_fit = model.fit(disp=True)

        y_test = test_ts[t:t + h]

        pred_results = model_fit.get_prediction(start=y_test.index[0],
                                                end=y_test.index[-1], dynamic=True)

        y_temp_pred = pred_results.predicted_mean

        y_pred = y_pred.append(y_temp_pred)

        history = history.append(y_test)

        print("%.2f percent complete..." % (100 * min((t + h), len(test_ts)) / len(test_ts)))

    mape = get_regression_score(test_ts.values, y_pred.values)

    title = '%d-%s Horizon Forecasting with Seasonal ARIMA (MAPE: %.2f%%)' \
            % (horizon * scale[1], scale[2], mape)
    ylabel = 'Traffic Volume x %s' % scale[0]

    plot_predictions(test_ts, y_pred, title, ylabel)
    plot_error_density(model_fit)

    return model_fit, mape, pd.Series(y_pred, index=test_ts.index)


def nstep_smoothing(ts, ratio, horizon, winlen, alpha, beta, gamma, scale):
    if ratio > 1 or ratio <= 0:
        print("\nWrong value of train ratio (train data/total data). Exiting!\n")
        return

    train_size = int(len(ts) * ratio)
    train_ts = ts[np.arange(train_size)]
    test_ts = ts[train_size:]
    history = train_ts
    y_pred = pd.Series()
    model_fit = None

    for t in range(0, len(test_ts), horizon):
        model = ExponentialSmoothing(history, freq=pd.infer_freq(ts.index),
                                     # trend='mul',
                                     seasonal='add', seasonal_periods=winlen
                                     )
        model_fit = model.fit(smoothing_level=alpha,
                              # smoothing_slope=beta,
                              smoothing_seasonal=gamma,
                              )
        yhat = model_fit.forecast(np.min([len(test_ts) - t, horizon]))

        history = history.append(test_ts[t:t + horizon])
        y_pred = y_pred.append(pd.Series(yhat.values,
                                         index=test_ts.index[t:t + horizon]))

        print("%.2f percent complete..." % (100 * min((t + horizon),
                                                      len(test_ts)) / len(test_ts)))

    mape = get_regression_score(test_ts.values, y_pred.values)

    title = '%d-%s Horizon Forecasting with Exponential Smoothing (MAPE: %.2f%%)' \
            % (horizon * scale[1], scale[2], mape)
    ylabel = 'Traffic Volume x %s' % scale[0]

    plot_predictions(test_ts, y_pred, title, ylabel)
    plot_error_density(model_fit)

    return model_fit, mape, pd.Series(y_pred, index=test_ts.index)


def plot_predictions(test, pred, title, ylabel):
    f = plt.figure()
    plt.plot(test, label='Original')
    plt.plot(pred, color='red', label='Predicted')
    plt.title(title, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()


def plot_error_density(model):
    g = plt.figure()

    plt.subplot(221)
    plt.plot(model.fittedvalues, model.resid, '.r')
    plt.title('Fitted Values vs Residual', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(222)
    sns.distplot(model.resid.values, kde=True, color='brown')
    plt.title('Density Plot of the Residual', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    ax = plt.subplot(223)
    qqplot(model.resid.values, ax=ax)
    ax.set_title('Normal Probability (Q-Q) Plot', fontsize=20)
    ax.set_xlabel('Theoretical Quantiles', fontsize=20)
    ax.set_ylabel('Residual Quantiles', fontsize=20)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), fontsize=15)
    ax.set_yticklabels(ax.yaxis.get_majorticklabels(), fontsize=15)

    plt.subplot(224)
    plt.stem(acf(model.resid))
    plt.title('Correlogram', color='k', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim([-1, 1.1])

    g.show()
