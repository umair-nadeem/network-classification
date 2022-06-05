import os
import itertools
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from networks.neural_utils import split_into_chunks, print_grid_results, mean_absolute_percentage_error
from series.series_utilities import get_regression_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/prediction_csv')


def xgb(ts, params, ratio, horizon, lag_depth, scale):
    booster = None

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    names = []
    for h in range(lag_depth + 1, 1, -1):
        name = 'f-' + str(h - 1)
        names.append(name)

    # Preparing the data matrices first
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=1 - ratio)

    x_temp_train = x_train
    y_temp_train = y_train
    y_pred = np.asarray([])

    for t in range(0, len(x_test), horizon):
        lgb_train = lgb.Dataset(x_temp_train, label=y_temp_train)

        booster = lgb.train(params, lgb_train, num_boost_round=100,
                            feature_name=names)

        x_horizon_test = x_test[t:t + horizon][:]
        y_horizon_test = y_test[t:t + horizon]

        x_temp = x_horizon_test[0]

        for h in range(np.min([len(x_test) - t, horizon])):
            y_forecast = booster.predict(x_temp.reshape(1, -1))

            x_temp = np.append(x_temp[1:], y_forecast)
            y_pred = np.append(y_pred, y_forecast)

        x_temp_train = np.append(x_train, x_horizon_test, axis=0)
        y_temp_train = np.append(y_train, y_horizon_test)

        print("%.2f percent complete..." % (100 * min((t + horizon), len(x_test)) / len(x_test)))

    print("\n\nThe prediction results for test data are:")
    mape = get_regression_score(y_test, y_pred)

    test_ind = np.where(ts == [y_test[0]])[0][0]
    y_pred = pd.Series(np.squeeze(np.asarray(y_pred)), index=ts.index[test_ind:])
    y_test = pd.Series(np.squeeze(np.asarray(y_test)), index=ts.index[test_ind:])

    title = '%d-%s Horizon Forecasting with Gradient Boosted Decision Trees (MAPE: %.2f%%)' \
            % (horizon * scale[1], scale[2], mape)
    ylabel = 'Traffic Volume x %s' % scale[0]
    f = plt.figure()
    plt.plot(y_test, label='Original')
    plt.plot(y_pred, color='red', label='Predicted')
    plt.title(title, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()

    name_list = np.asarray([])
    for i in range(lag_depth, 0, -1):
        name_list = np.append(name_list, 'f-' + str(i))

    importance = pd.Series(booster.feature_importance('split'), index=name_list)
    importance = importance.sort_values(ascending=True)

    g = plt.figure()
    plt.barh(importance.index[-20:], importance.values[-20:])
    plt.title('GBT Feature Importance', fontsize=30)
    plt.xlabel('Feature Importance', fontsize=25)
    plt.ylabel('Features', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    g.show()

    return booster, mape, y_pred


def forest(ts, params, ratio, horizon, lag_depth, scale):
    model = None

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    # Preparing the data matrices first
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=1 - ratio)

    x_temp_train = x_train
    y_temp_train = y_train
    y_pred = np.asarray([])

    for t in range(0, len(x_test), horizon):
        model = RandomForestRegressor(n_estimators=params['n_estimators'],
                                      criterion=params['criterion'],
                                      # max_depth=params['max_depth'],
                                      # min_samples_leaf=params['min_samples_leaf'],
                                      # max_leaf_nodes=params['max_leaf_nodes'],
                                      n_jobs=params['n_jobs'],
                                      verbose=params['verbose'])

        model.fit(x_temp_train, y_temp_train)

        x_horizon_test = x_test[t:t + horizon][:]
        y_horizon_test = y_test[t:t + horizon]

        x_temp = x_horizon_test[0]

        for h in range(np.min([len(x_test) - t, horizon])):
            y_forecast = model.predict(x_temp.reshape(1, -1))

            x_temp = np.append(x_temp[1:], y_forecast)
            y_pred = np.append(y_pred, y_forecast)

        x_temp_train = np.append(x_train, x_horizon_test, axis=0)
        y_temp_train = np.append(y_train, y_horizon_test)

        print("%.2f percent complete..." % (100 * min((t + horizon), len(x_test)) / len(x_test)))

    print("\n\nThe prediction results for test data are:")
    mape = get_regression_score(y_test, y_pred)

    test_ind = np.where(ts == [y_test[0]])[0][0]
    y_pred = pd.Series(np.squeeze(np.asarray(y_pred)), index=ts.index[test_ind:])
    y_test = pd.Series(np.squeeze(np.asarray(y_test)), index=ts.index[test_ind:])

    title = '%d-%s Horizon Forecasting with Random Forests (MAPE: %.2f%%)' \
            % (horizon * scale[1], scale[2], mape)
    ylabel = 'Traffic Volume x %s' % scale[0]
    f = plt.figure()
    plt.plot(y_test, label='Original')
    plt.plot(y_pred, color='red', label='Predicted')
    plt.title(title, fontsize=30)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()

    name_list = np.asarray([])
    for i in range(lag_depth, 0, -1):
        name_list = np.append(name_list, 'f-' + str(i))

    importance = pd.Series(model.feature_importances_, index=name_list)
    importance = importance.sort_values(ascending=False)

    # Plot the feature importances of the forest
    g = plt.figure()
    plt.bar(importance.index[:20], importance.values[:20],
            color="r", align="center")
    plt.title("Random Forest Feature Importance (%)", fontsize=30)
    plt.xlabel('Features', fontsize=25)
    plt.ylabel('Feature Importance', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    g.show()

    return model, mape, y_pred


def forest_search(ts, params, ratio, lag_depth):
    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=1 - ratio)

    mape = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    param_grid = {'n_estimators': [40, 80, 120, 160, 200],  # Gaussian mixture
                  'max_depth': [10, 12, 15],
                  'min_samples_leaf': [1, 5, 10, 20],
                  'max_leaf_nodes': [5, 10]}

    model = RandomForestRegressor(criterion=params['criterion'],
                                  max_leaf_nodes=params['max_leaf_nodes'])

    clf = GridSearchCV(estimator=model, param_grid=param_grid,
                       scoring=mape, cv=3, return_train_score=True,
                       verbose=2, n_jobs=-1)

    grid_result = clf.fit(x_train, y_train)

    print_grid_results(grid_result)

    del model

    return clf
