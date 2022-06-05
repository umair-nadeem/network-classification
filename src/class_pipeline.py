import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor

from networks.neural_utils import print_grid_results
from networks.netmodel import neural_classification, grid_search_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/classification_csv')


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.magma_r):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    g = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.title(title, fontsize=30)
    plt.ylabel('True Class', fontsize=25)
    plt.xlabel('Predicted Class', fontsize=25)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=15)
    plt.tight_layout()
    g.show()


def naivebayes():
    model = GaussianNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=model.classes_,
                          title='Naive Bayes Confusion Heatmap')


def svc():
    model = SVC(C=0.1,
                kernel='linear',
                verbose=1,
                decision_function_shape='ovr',
                max_iter=-1)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=model.classes_,
                          title='SVM Confusion Heatmap')


def lsvc():
    model = LinearSVC(C=2,
                      loss='hinge',
                      verbose=1,
                      max_iter=1000,
                      multi_class='ovr')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(cnf_matrix, classes=model.classes_,
                          title='SVM Confusion Heatmap')


def svc_cv():
    param_grid = {'C': [0.1, 1.0, 5.0, 10]}

    clf = GridSearchCV(SVC(kernel='linear',
                           verbose=1,
                           decision_function_shape='ovr',
                           max_iter=-1), param_grid=param_grid,
                       scoring='accuracy', n_jobs=-1,
                       cv=3, verbose=2, return_train_score=True)

    grid_result = clf.fit(x_train, y_train)

    print_grid_results(grid_result)

    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))


# Reading the train and test datasets
x_train = pd.read_csv('train.csv')
x_test = pd.read_csv('test.csv')

# Reading the train and test source datasets
train_source = pd.read_csv('source_train.csv')
test_source = pd.read_csv('source_test.csv')

# Reading the train and test source datasets
train_destination = pd.read_csv('destination_train.csv')
test_destination = pd.read_csv('destination_test.csv')

# Extracting the labels
y_train = x_train['appName'].values.astype(str)
y_test = x_test['appName'].values

# One-hot transforming the protocols
lb = LabelBinarizer()
train_protocol = lb.fit_transform(x_train['protocolName'])
test_protocol = lb.transform(x_test['protocolName'])

x_train = np.asarray(x_train.drop(columns=['source',
                                           'destination',
                                           'protocolName',
                                           'appName']))

x_test = np.asarray(x_test.drop(columns=['source',
                                         'destination',
                                         'protocolName',
                                         'appName']))

x_train = np.append(x_train, train_protocol, axis=1)
x_test = np.append(x_test, test_protocol, axis=1)

x_train = np.append(x_train, np.asarray(train_source), axis=1)
x_test = np.append(x_test, np.asarray(test_source), axis=1)

x_train = np.append(x_train, np.asarray(train_destination), axis=1)
x_test = np.append(x_test, np.asarray(test_destination), axis=1)

"""
history = neural_classification(batch_size=100, epochs=10,
                                neurons=[20, 40, 18],
                                activation='tanh',
                                activation_output='sigmoid',
                                initializer='lecun_normal',
                                loss='binary_crossentropy',
                                optimizer='Adam', dropout_rate=0.0,
                                weight_constraint=0.95,
                                x_train=x_train, y_train=y_train,
                                x_test=x_test, y_test=y_test)

"""
