import os
import itertools
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split

from networks.neural_utils import split_into_chunks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('seaborn')
os.chdir('/home/umair/PycharmProjects/thesis/files/prediction_csv')

TENSORBOARD_DIR = '/home/umair/PycharmProjects/thesis/files/graphs'

DATASET = 'short.csv'

ratio = 0.667
LEARN_RATE = 0.001
batch_size = 80
lag_depth = 168
horizon = 24

df = pd.read_csv(DATASET, names=['time', 'len'], dtype=float, parse_dates=['time'])
df = df.set_index('time')
ts = df.squeeze()

[X, Y] = split_into_chunks(ts.values, lag_depth)
X = np.asarray(X)
Y = np.asarray(Y)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, shuffle=True, test_size=1 - ratio)

x_temp_train = x_train
y_temp_train = y_train
y_pred = np.asarray([])

history = np.asarray([])
val_history = np.asarray([])


def dense_layer(dense_input, neurons, activation, name='dense'):
    return tf.layers.dense(dense_input, units=neurons, activation=activation,
                           use_bias=True, kernel_initializer=tf.glorot_normal_initializer(),
                           bias_initializer=tf.zeros_initializer(),
                           trainable=True, name=name, reuse=False)


def mlp(x, weights, biases):
    layer1_mult = tf.matmul(x, weights['w1'], name='layer1_mult')
    layer1_out = tf.add(layer1_mult, biases['b1'], name='layer1_out')
    layer1_act = tf.nn.elu(layer1_out, name='relu')

    tf.summary.histogram("ReLU1", layer1_act)

    layer2_mult = tf.matmul(layer1_act, weights['w2'], name='layer2_mult')
    layer2_out = tf.add(layer2_mult, biases['b2'], name='layer2_out')
    return layer2_out


# Store layers weight & bias
weights = {
    'w1': tf.get_variable(name='W1', shape=[X.shape[1], 4],
                          initializer=tf.glorot_normal_initializer()),
    'w2': tf.get_variable(name='W2', shape=[4, 1],
                          initializer=tf.glorot_normal_initializer()),
}
biases = {
    'b1': tf.get_variable(name='b1', shape=[4],
                          initializer=tf.zeros_initializer()),
    'b2': tf.get_variable(name='b2', shape=[1],
                          initializer=tf.zeros_initializer()),
}

x = tf.placeholder(dtype=tf.float32, shape=[None, X.shape[1]], name='X')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='Y')

output = mlp(x, weights, biases)

sess = tf.InteractiveSession()

with tf.name_scope('MAPE'):
    mape = tf.abs(tf.subtract(y, output)) / tf.abs(y)
    loss = tf.reduce_mean(mape) * 100

with tf.name_scope('Adamax'):
    # Optimizer operations
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARN_RATE)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    optimizer_min = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)

# collecting summaries of all the weights and biases
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)

for grad, var in grads:
    tf.summary.histogram(var.name + '/gradient', grad)

merged_summary = tf.summary.merge_all()
filewriter = tf.summary.FileWriter(TENSORBOARD_DIR, sess.graph)

sess.run(tf.global_variables_initializer())

# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
for step in range(0, len(x_train), batch_size):
    batch_data = x_train[step:(step + batch_size), :]
    batch_labels = y_train[step:(step + batch_size)]
    batch_labels = np.expand_dims(batch_labels, axis=1)

    feed_dict = {x: batch_data, y: batch_labels}

    out, summary, _, cost = sess.run([output, merged_summary,
                                      optimizer_min, loss],
                                     feed_dict=feed_dict)

    filewriter.add_summary(summary, step)

    print('Step %d: Minibatch loss: %0.3f\n' % (step, cost))

filewriter.close()
