import os

from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_normal
from keras.wrappers.scikit_learn import KerasRegressor

from .neural_utils import do_search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(activation='relu'):
    model = Sequential()

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    activation=activation, kernel_initializer=initializer))

    for i in range(1, len(neurons) - 1):
        model.add(Dense(neurons[i], activation=activation, kernel_initializer=initializer))

    model.add(Dense(neurons[-1], activation='linear', kernel_initializer=initializer))

    model.compile(loss=loss, optimizer=optimizer)

    return model


def search_activation_grid(batch_size, epochs, ts,
                           ratio, lag, all_neurons, init,
                           metric, opt):
    global lag_depth
    global neurons
    global initializer
    global loss
    global optimizer

    lag_depth = lag
    neurons = all_neurons
    initializer = init
    loss = metric
    optimizer = opt

    activation = ['softmax', 'softplus', 'softsign', 'elu', 'selu',
                  'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    param_grid = dict(activation=activation)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    # passing over the values to start grid search
    return do_search(keras_model, param_grid, ts, ratio, lag_depth)
