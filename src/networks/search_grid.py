import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import split_into_chunks, mean_absolute_percentage_error, print_grid_results


def create_model(activation='relu', optimizer='adam'):
    model = Sequential()

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation=activation, kernel_initializer=initializer))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation=activation, kernel_initializer=initializer))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=initializer))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def grid_search(batch_size, epochs, ts,
                ratio, lag, all_neurons, init,
                metric):
    global lag_depth
    global neurons
    global loss
    global initializer

    lag_depth = lag
    neurons = all_neurons
    loss = metric
    initializer = init

    activation = ['softplus', 'elu', 'relu', 'linear']

    optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

    param_grid = dict(activation=activation, optimizer=optimizer)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=abs(1 - ratio))

    mape = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

    clf = GridSearchCV(estimator=keras_model, param_grid=param_grid,
                       scoring=mape, cv=5, verbose=1, n_jobs=-1)

    grid_result = clf.fit(x_train, y_train, shuffle=False)

    print_grid_results(grid_result)
