import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import keras.backend as K

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

plt.style.use('seaborn')


def dickeyfuller_stationarity(series):
    """Augmented Dickey Fuller test is used to detect unit
    root in time series. It does hypothesis testing to reject
    or accept the null hypothesis, in this case, that unit
    root is present. Unit root inidcates presence of trend
    and/or seasonality in time series data.

    The significance of hypothesis test is quantified via
    p-value. Given that the tested data sample is true
    representation of observed data, the p-value is the
    probability of occurence of more extreme value under
    the null hypothesis. p-value gives the confidence that
    the tested data satisfies the null hypothesis.

    The ADF test statistic is a negative number, the larger
    the value, the stronger the rejection of null hypothesis."""

    # Perform Dickey-Fuller test:
    dftest = adfuller(series, autolag='AIC')
    keys = ['Test Statistic', 'p-value', '#Lags Used', '# Observations']
    result = list(dftest[4].items())
    result.extend(list(zip(keys, dftest[0:4])))

    for key, value in result:
        print('{0:>15} {1} {2} {3:.5e}'.format(key, ':', ' ', value))


def differenced_stationarity(data):
    data_diff = data - data.shift()
    data_diff.dropna(inplace=True)
    dickeyfuller_stationarity(data_diff)


def decomposition_stationarity(data, winlen, name):
    decomp = seasonal_decompose(data, freq=winlen)
    trend = decomp.trend
    seasonal = decomp.seasonal
    resid = decomp.resid
    resid.dropna(inplace=True)
    dickeyfuller_stationarity(resid)

    plt.figure()
    plt.subplot(411)
    plt.plot(data, label='Original')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Seasonal Decomposition of %s Time Series' % name, fontsize=30)
    plt.legend(loc='best', fontsize=15)
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.subplot(414)
    plt.plot(resid, label='Residual')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.tight_layout()
    plt.show()


def get_lag_plots(data, n, name):
    x = np.array(data.values)
    x = np.asmatrix(x)

    for i in range(0, n):
        shifted = data.shift(i + 1)
        x = np.insert(x, i + 1, shifted, axis=0)
        plt.figure()
        plt.plot(x[(i + 1), :], x[0, :], '*')
        plt.title("%d Shift Lag Plot of %s Time Series" % (i + 1, name), fontsize=30)
        plt.xlabel('Original series', fontsize=25)
        plt.ylabel('Shifted series', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()


def plot_data(data, name, scale):
    ylabel = 'Traffic Volume (x%s)' % scale[0]
    plt.figure()
    plt.plot(data)
    plt.title("%s Time Series" % name, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def plot_diff_data(data, name, scale):
    data_diff = data - data.shift()
    data_diff.dropna(inplace=True)

    ylabel = 'Traffic Volume (x%s)' % scale[0]
    plt.figure()
    plt.plot(data_diff)
    plt.title("%s Time Series (Differenced)" % name, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


def plot_acf_pacf(data, lags):
    data_pacf = pacf(data, nlags=20)
    data_pacf[0] = 0

    plot_acf(data, lags=lags)
    plt.title('ACF', fontsize=30)
    plt.xlabel('Lag', fontsize=25)
    plt.ylabel('Correlation Coefficient', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.stem(range(len(data_pacf)), data_pacf)
    ax1.axhline(y=1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g',
                label='90% Confidence Interval')
    ax1.axhline(y=-1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g')
    ax1.axhline(y=1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r',
                label='95% Confidence Interval')
    ax1.axhline(y=-1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r')
    ax1.axhline(y=2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black',
                label='99% Confidence Interval')
    ax1.axhline(y=-2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black')
    ax1.set_title('PACF', fontsize=30)
    ax1.set_xlabel('Lag', fontsize=25)
    ax1.set_ylabel('Correlation Coefficient', fontsize=25)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.legend(fontsize=15)
    plt.grid(True)
    plt.show()


def plot_diff_acf_pacf(data):
    data_diff = data - data.shift()
    data_diff.dropna(inplace=True)

    data_acf = acf(data_diff, nlags=20)
    data_pacf = pacf(data_diff, nlags=20)
    data_acf[0] = 0
    data_pacf[0] = 0

    fig, ax = plt.subplots()
    ax.stem(range(len(data_acf)), data_acf)
    ax.axhline(y=1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g',
               label='90% Confidence Interval')
    ax.axhline(y=-1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g')
    ax.axhline(y=1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r',
               label='95% Confidence Interval')
    ax.axhline(y=-1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r')
    ax.axhline(y=2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black',
               label='99% Confidence Interval')
    ax.axhline(y=-2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black')
    ax.set_title('ACF of Differenced Series', fontsize=30)
    ax.set_xlabel('Lag', fontsize=25)
    ax.set_ylabel('Correlation Coefficient', fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.legend(fontsize=15)
    ax.grid(True)

    fig, ax1 = plt.subplots()
    ax1.stem(range(len(data_pacf)), data_pacf)
    ax1.axhline(y=1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g',
                label='90% Confidence Interval')
    ax1.axhline(y=-1.645 / np.sqrt(len(data)), linewidth=1, marker='.', color='g')
    ax1.axhline(y=1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r',
                label='95% Confidence Interval')
    ax1.axhline(y=-1.96 / np.sqrt(len(data)), linewidth=1, marker='.', color='r')
    ax1.axhline(y=2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black',
                label='99% Confidence Interval')
    ax1.axhline(y=-2.576 / np.sqrt(len(data)), linewidth=1, marker='.', color='black')
    ax1.set_title('PACF of Differenced Series', fontsize=30)
    ax1.set_xlabel('Lag', fontsize=25)
    ax1.set_ylabel('Correlation Coefficient', fontsize=25)
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.legend(fontsize=15)
    plt.grid(True)
    plt.show()


def get_regression_score(y_true, y_pred):
    """This method receives numpy arrays y_true and y_pred
    and calculates three performance metrics i.e. mean
    squared error, mean absolute error and mean absolute
    percentage error."""
    y_true[np.where(y_true == 0.0)] = 0.001
    mape = np.mean(np.abs((y_true - y_pred) / np.abs(y_true))) * 100.

    print('\nMean Absolute Percentange Error (MAPE): %.2f\n' % mape)

    return mape
