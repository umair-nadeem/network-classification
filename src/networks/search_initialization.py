import os

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import do_search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(initializer='uniform'):
    model = Sequential()

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation=activation, kernel_initializer=initializer))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation=activation, kernel_initializer=initializer))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=initializer))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def search_initialization_grid(batch_size, epochs, ts,
                               ratio, lag, all_neurons, act,
                               metric, opt):
    global lag_depth
    global neurons
    global activation
    global loss
    global optimizer

    lag_depth = lag
    neurons = all_neurons
    activation = act
    loss = metric
    optimizer = opt

    initializer = ['Zeros', 'Ones',
                   'RandomNormal', 'RandomUniform',
                   'glorot_normal', 'glorot_uniform',
                   'lecun_normal', 'lecun_uniform',
                   'he_normal', 'he_uniform']

    param_grid = dict(initializer=initializer)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    # passing over the values to start grid search
    return do_search(keras_model, param_grid, ts, ratio, lag_depth)
