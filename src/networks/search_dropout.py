import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.initializers import glorot_normal
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import do_search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(dropout_rate=0.0, weight_constraint=0):
    model = Sequential()

    gn = glorot_normal(seed=(2 / (lag_depth + 1)) ** (1 / 2))

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation='relu', kernel_initializer=gn,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation='relu', kernel_initializer=gn,
                        kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=gn))

    model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

    return model


def search_dropout_grid(batch_size, epochs, ts,
                        ratio, lag, all_neurons):
    if len(all_neurons) < 3:
        print("\nNot enough neurons for defining model layers. Exiting!\n")
        return

    global lag_depth
    global neurons
    lag_depth = lag
    neurons = all_neurons

    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    # passing over the values to start grid search
    return do_search(keras_model, param_grid, ts, ratio, lag_depth)
