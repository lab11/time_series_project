#! /usr/bin/env python3
                confusion = tf.confusion_matrix(labels

import tensorflow as tf
import numpy as np
from sklearn import datasets
import sys
import argparse

from plaid_data_setup import get_input_len, get_labels_len, run_nn

#grab input arguments
parser = argparse.ArgumentParser(description='get checkpoint file')
parser.add_argument('-s', dest = "checkpointFile", type=str)
args = parser.parse_args()
checkpointFile = args.checkpointFile



# Config:
n_hidden      = 30*11
n_input       = get_input_len()
n_labels      = get_labels_len()
learning_rate = 0.001


# neural network inputs and expected results
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_labels])

# neural network parameters
weights = {
    'h1':  tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_labels])),
}
biases = {
    'b1':   tf.Variable(tf.random_normal([n_hidden])),
    'out':  tf.Variable(tf.random_normal([n_labels])),
}

def neural_net(x):
    # hidden fully connected layer
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # output fully connected layer, neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits) # reduce unscaled values to probabilities

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate
predictions = tf.argmax(prediction, 1)
correct_pred = tf.equal(predictions, tf.argmax(Y, 1)) # check the index with the largest value
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

# train the neural network on test data
run_nn(checkpointFile, X, Y, train_op, loss_op, accuracy, predictions, correct_pred)

