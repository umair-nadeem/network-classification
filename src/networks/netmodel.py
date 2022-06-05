import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers

from networks.neural_utils import split_into_chunks, print_grid_results
from series.series_utilities import get_regression_score


def neural_prediction(batch_size, epochs, ts, ratio, horizon, lag_depth, neurons,
                      activation, initializer, loss, optimizer, dropout_rate,
                      weight_constraint, scale):
    """Implements a keras deep learning model with given parameters and plots
    and prints prediction results for test data."""

    [x, y] = split_into_chunks(ts.values, lag_depth)
    x = np.asarray(x)
    y = np.asarray(y)

    # Preparing the data matrices first
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, shuffle=False, test_size=1 - ratio)

    x_temp_train = x_train
    y_temp_train = y_train
    y_pred = np.asarray([])

    history = np.asarray([])
    val_history = np.asarray([])

    # constructing the keras model
    model = Sequential()

    model.add(Dense(neurons[0], input_dim=lag_depth,
                    # activity_regularizer=regularizers.l2(0.02),
                    activation=activation, kernel_initializer=initializer,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))

    # adding hidden layers
    for neuron in neurons[1:-1]:
        model.add(Dense(neuron, activation=activation, kernel_initializer=initializer,
                        kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(neurons[-1], activation='linear',
                    # activity_regularizer=regularizers.l2(0.02),
                    kernel_initializer=initializer))

    model.compile(loss=loss, optimizer=optimizer)

    for t in range(0, len(x_test), horizon):

        train_history = model.fit(x_temp_train, y_temp_train, epochs=epochs,
                                  callbacks=[EarlyStopping(monitor='loss', min_delta=0.8,
                                                           patience=5, verbose=0, mode='min')],
                                  # validation_split=1 - ratio,
                                  batch_size=batch_size, verbose=0, shuffle=False)

        history = np.append(history, train_history.history['loss'])
        # val_history = np.append(val_history, train_history.history['val_loss'])

        x_horizon_test = x_test[t:t + horizon][:]
        y_horizon_test = y_test[t:t + horizon]

        x_temp = x_horizon_test[0]

        for h in range(np.min([len(x_test) - t, horizon])):
            y_forecast = model.predict(x_temp.reshape(1, -1),
                                       batch_size=batch_size).reshape(-1)

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

    title = '%d-%s Horizon Forecasting with Neural Network (MAPE: %.2f%%)' \
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

    g = plt.figure()
    plt.plot(history, label='Training Error')
    # plt.plot(val_history, label='Cross Validation Error', color='brown')
    plt.title('Training Error vs Iterations', fontsize=30)
    plt.xlabel('Iterations', fontsize=25)
    plt.ylabel('Mean Absolute Percentage Error', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    g.show()

    return history, mape, y_pred


def neural_classification(batch_size, epochs, neurons, activation,
                          activation_output, initializer, loss,
                          optimizer, dropout_rate, weight_constraint,
                          x_train, y_train, x_test, y_test):
    """this methods create the neural network with given params
    to do the classification task"""

    x_train = np.append(x_train, x_test, axis=0)
    y_train = np.append(y_train, y_test, axis=0)

    label_lb = LabelBinarizer()
    label_y_train = label_lb.fit_transform(y_train)

    model = Sequential()

    model.add(Dense(neurons[0], input_dim=x_train.shape[1],
                    # activity_regularizer=regularizers.l2(0.02),
                    activation=activation, kernel_initializer=initializer,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))

    # adding hidden layers
    for neuron in neurons[1:-1]:
        model.add(Dense(neuron, activation=activation, kernel_initializer=initializer,
                        kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))

    model.add(Dense(neurons[-1], activation=activation_output,
                    # activity_regularizer=regularizers.l2(0.02),
                    kernel_initializer=initializer))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    train_history = model.fit(x_train, label_y_train, epochs=epochs,
                              callbacks=[EarlyStopping(monitor='loss', min_delta=2,
                                                       patience=5, verbose=0, mode='min')],
                              # validation_split=0.25,
                              batch_size=batch_size, verbose=0, shuffle=False)

    label_y_pred = model.predict(x_test)
    y_pred = label_lb.inverse_transform(label_y_pred)

    print(classification_report(y_test, y_pred))
    print()
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    f = plt.figure()
    plt.subplot(212)
    plt.plot(train_history.history['acc'], label='Accuracy', color='g')
    # plt.plot(train_history.history['val_acc'], label='Accuracy', color='black')
    plt.title('Training Accuracy', fontsize=30)

    plt.subplot(211)
    # plt.plot(train_history.history['val_loss'], label='CV Loss', color='brown')
    plt.plot(train_history.history['loss'], label='Loss', color='r')

    plt.title('Training Loss', fontsize=30)
    plt.legend(loc='best', fontsize=15)
    plt.grid(True)
    f.show()

    return train_history


def create_model(activation='tanh', optimizer='adam'):
    neurons = [20, 30, 50, 18]

    model = Sequential()

    model.add(Dense(neurons[0], input_dim=667,
                    activation=activation, kernel_initializer='glorot_normal'))

    for neuron in neurons[1:-1]:
        model.add(Dense(neuron, activation=activation, kernel_initializer='glorot_normal'))

    model.add(Dense(neurons[-1], activation='softmax', kernel_initializer='glorot_normal'))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def grid_search_classification(batch_size, epochs,
                               x_train, y_train, x_test, y_test):
    activation = ['softplus', 'softmax', 'tanh', 'sigmoid']

    optimizer = ['RMSprop', 'Adam', 'Adamax', 'Adagrad', 'Adadelta', 'Nadam']

    label_lb = LabelBinarizer()
    label_y_train = label_lb.fit_transform(y_train)

    param_grid = dict(activation=activation, optimizer=optimizer)

    keras_model = KerasRegressor(build_fn=create_model, epochs=epochs,
                                 batch_size=batch_size, verbose=0)

    clf = GridSearchCV(estimator=keras_model, param_grid=param_grid,
                       scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

    grid_result = clf.fit(x_train, label_y_train)

    print_grid_results(grid_result)

    label_y_pred = clf.predict(x_test)
    y_pred = label_lb.inverse_transform(label_y_pred)

    print(classification_report(y_test, y_pred))
