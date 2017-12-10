#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
import os
import itertools

from barsim_data_setup import gen_data, run_nn, get_input_len, get_labels_len


# Config:
n_hidden      = 30
n_input       = get_input_len()
n_outputs     = 2
n_labels      = get_labels_len()
n_networks    = len(list(itertools.combinations(range(n_labels), n_outputs)))
learning_rate = 0.001

# arrays of neural network hooks
tf_inputs = []
tf_expecteds = []
train_ops = []
loss_ops = []
accuracy_ops = []

# create neural network (mostly as per Barsim et. al)
def neural_net():
    # neural network inputs
    tf_input = tf.placeholder("float", [None, n_input])
    tf_expected = tf.placeholder("float", [None, n_outputs])

    # neural network parameters
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_outputs])),
    }
    biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_outputs])),
    }

    # hidden fully-connected layer
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(tf_input, weights['h1']), biases['h1']))

    # possibly add dropout here? the paper doesn't say anything about dropout

    # output fully-connected layer
    out_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

    # loss operation, unclear if this is what the paper does
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=tf_expected))

    # training, using Adam optimzer rather than conjugate gradient descent because that's not in tensor flow
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # accuracy, unclear if this is equivalent to what they do
    predictions = tf.nn.softmax(out_layer)
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(tf_expected, 1)) # check the index with the largest value
    accuracy_op = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) # percentage of traces that were correct

    # save hooks into neural network for training and evaluation
    tf_inputs.append(tf_input)
    tf_expecteds.append(tf_expected)
    train_ops.append(train_op)
    loss_ops.append(loss_op)
    accuracy_ops.append(accuracy_op)

# create all combinations of binary classifier neural networks
print("Creating neural networks")
for index in range(n_networks):
    neural_net()

# train the neural network ensembles on test data
print("Beginning training")
run_nn(tf_inputs, tf_expecteds, train_ops, loss_ops, accuracy_ops, gen_data())

