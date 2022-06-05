import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import glorot_normal
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import do_search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(learn_rate=0.01, momentum=0):
    model = Sequential()

    gn = glorot_normal(seed=(2 / (lag_depth + 1)) ** (1 / 2))

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation='relu', kernel_initializer=gn))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation='relu', kernel_initializer=gn))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=gn))

    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mean_absolute_percentage_error', optimizer=optimizer)

    return model


def search_rate_momentum_grid(batch_size, epochs, ts,
                              ratio, lag, all_neurons):
    if len(all_neurons) < 3:
        print("\nNot enough neurons for defining model layers. Exiting!\n")
        return

    global lag_depth
    global neurons
    lag_depth = lag
    neurons = all_neurons

    learn_rate = [0.001, 0.01, 0.1, 0.2]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8]

    param_grid = dict(learn_rate=learn_rate, momentum=momentum)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    # passing over the values to start grid search
    return do_search(keras_model, param_grid, ts, ratio, lag_depth)
