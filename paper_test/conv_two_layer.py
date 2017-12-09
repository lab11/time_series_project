#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sys

from plaid_data_setup import run_nn, generate_training_and_validation

# function to create the training and validation datasets
def gen_data():
    # load and shuffle data
    data = np.load("../plaid_data/traces_bundle.npy")
    np.random.shuffle(data)
    Data = data[:, 0:-2]

    # normalize all waveform magnitude to the maximum for that type
    data_len = len(Data[0])
    Data[:, :data_len/2] /= np.amax(np.absolute(Data[:, :data_len/2])) # current
    Data[:, data_len/2:] /= np.amax(np.absolute(Data[:, data_len/2:])) # voltage

    Data = np.reshape(Data, (Data.shape[0], int(Data.shape[1]/2), 2), 'F')
    Labels = data[:,-1]
    Names = data[:,-2]

    # get label string names and pad spaces to make them equal length
    labelstrs = np.load("../plaid_data/traces_class_map.npy")
    max_str_len = max([len(s) for s in labelstrs])
    for index, label in enumerate(labelstrs):
        labelstrs[index] = label + ' '*(max_str_len - len(label))

    # quick idiot test
    if max(Labels)+1 != len(labelstrs):
        print("Error: Number of classes doesn't match labels input")
        sys.exit()

    # generate training and validation datasets (already shuffled)
    TrainingData, TrainingLabels, ValidationData, ValidationLabels = generate_training_and_validation(Data, Labels, Names, 0.20)

    return (TrainingData, ValidationData, TrainingLabels, ValidationLabels, labelstrs)

def get_input_len():
    # length of data dimension, minus 2 (label and name)
    return int((np.shape(np.load("../plaid_data/traces_bundle.npy"))[1] - 2)/2)

def get_labels_len():
    # number of classes saved
    return np.shape(np.load("../plaid_data/traces_class_map.npy"))[0]

# Config:
conv_filt_size = 20
n_conv_filts = 3
n_hidden    = 100
n_input     = get_input_len()
n_labels    = get_labels_len()
learning_rate = 0.001
drop_probability = 0#0.50


# neural network inputs and expected results
X = tf.placeholder("float", [None, n_input, 2])
Y = tf.placeholder("float", [None, n_labels])

# neural network parameters
weights = {
    'conv': tf.Variable(tf.random_normal([conv_filt_size, 2, 1, n_conv_filts])),
    'h1':  tf.Variable(tf.random_normal([n_input*n_conv_filts, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_labels])),
}
biases = {
    'conv': tf.Variable(tf.random_normal([n_conv_filts])),
    'b1':   tf.Variable(tf.random_normal([n_hidden])),
    'out':  tf.Variable(tf.random_normal([n_labels])),
}

def neural_net(x):
    #reshape for input to the convolution
    rdata = tf.reshape(x,[-1,n_input,2,1])
    # a small convolutional layer to learn filters
    conv_1 = tf.nn.relu(tf.nn.conv2d(rdata,weights['conv'], strides=[1,1,2,1], padding='SAME') + biases['conv'])
    #reshape the conv output to be flat
    conv_1_out = tf.reshape(conv_1, [-1,n_input*n_conv_filts])
    # hidden fully connected layer
    layer_1 = tf.nn.relu(tf.add(tf.matmul(conv_1_out, weights['h1']), biases['b1']))
    # add some dropout
    layer_1_drop = tf.nn.dropout(layer_1, 1.0 - drop_probability)
    # output fully connected layer, neuron for each class
    out_layer = tf.matmul(layer_1_drop, weights['out']) + biases['out']
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
run_nn(X, Y, train_op, loss_op, accuracy, predictions, correct_pred, gen_data())


