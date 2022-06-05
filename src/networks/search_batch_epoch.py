import os

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_normal
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import do_search, split_into_chunks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model():
    model = Sequential()

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation=activation, kernel_initializer=initializer))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation=activation, kernel_initializer=initializer))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=initializer))
    model.compile(loss=loss, optimizer=optimizer)

    return model


def search_batch_epoch_grid(batch_values, epoch_values, ts,
                            ratio, lag, all_neurons, act, init,
                            metric, opt):
    """This method takes sequences of batch_size and epochs values
    and passes them as a parameter grid to Keras' scikit-learn's
    wrapper class of GridSearchCV.

    The global variables are overwritten by passed values.

    The do_search method of utils module performs train-test
    splitting, scaling, normalization, grid search and results
    manipulation."""

    global lag_depth
    global neurons
    global activation
    global initializer
    global loss
    global optimizer

    lag_depth = lag
    neurons = all_neurons
    activation = act
    initializer = init
    loss = metric
    optimizer = opt

    param_grid = dict(batch_size=batch_values, epochs=epoch_values)

    keras_model = KerasRegressor(build_fn=create_model, verbose=0)

    # passing over the values to start grid search
    return do_search(keras_model, param_grid, ts, ratio, lag_depth)
