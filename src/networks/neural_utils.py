import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from series.series_utilities import get_regression_score


def do_search(model, param_grid, ts, test_size, lag_depth):
    """Method for carrying out hyper parameter grid search
    with scikit-learn's class GridSearchCV.

    It takes in the data matrix x, labels y, sklearn estimator
    or keras estimator with keras' scikit-learn wrapper."""

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=abs(1 - test_size))

    mape = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    clf = GridSearchCV(estimator=model, param_grid=param_grid,
                       scoring=mape, cv=3, verbose=1, n_jobs=-1)

    grid_result = clf.fit(x_train, y_train)

    print_grid_results(grid_result)

    del model

    return grid_result


def mean_absolute_percentage_error(y_true, y_pred):
    # y_true[np.where(y_true == 0.0)] = 0.001
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.


def print_grid_results(grid_result):
    print("\nBest score: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def scale_data(train, test):
    # Transforming and scaling the data matrix i.e. 0 mean & unit variance
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test


def split_into_chunks(data, depth):
    x, y = [], []
    for i in range(0, len(data), 1):
        try:
            x_i = data[i:i + depth]
            y_i = data[i + depth]

        except IndexError:
            break
        x.append(x_i)
        y.append(y_i)
    return x, y
