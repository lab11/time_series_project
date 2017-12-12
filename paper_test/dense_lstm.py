#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
import sys

from plaid_data_setup import get_input_len, get_labels_len, train_device_cycle_nn, gen_data


def build_nn():

    graph = tf.Graph()
    with graph.as_default():

        # Config:
        n_hidden         = 30*11
        n_dense_out      = 20
        n_lstm_units     = 12
        n_input          = get_input_len()
        n_labels         = get_labels_len()
        learning_rate    = 0.001

        inputs = tf.placeholder(tf.float32, (None, None, n_input))
        correct_labels = tf.placeholder(tf.float32, (None, n_labels))
        # The true sequence lengths of the inputs. The actual inputs
        # should be zero-padded such that they are all the same length.
        seqlen = tf.placeholder(tf.int32, None)
        dropout_prob = tf.placeholder_with_default(1.0, shape=())

        # neural network parameters
        weights = {
            'h1':  tf.Variable(tf.random_normal([n_input, n_hidden])),
            'dense_out':  tf.Variable(tf.random_normal([n_hidden, n_dense_out])),
            'lstm_out': tf.Variable(tf.random_normal([n_lstm_units, n_labels])),
        }
        biases = {
            'b1':   tf.Variable(tf.random_normal([n_hidden])),
            'dense_out':  tf.Variable(tf.random_normal([n_dense_out])),
            'lstm_out':  tf.Variable(tf.random_normal([n_labels])),
        }


        def neural_net(x):
            # hidden fully connected layer
            tweight = tf.tile(tf.expand_dims(weights['h1'], axis=0),(tf.shape(x)[0], 1, 1))
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, tweight), biases['b1']))
            # dropout on hidden layer
            layer_1_drop = tf.nn.dropout(layer_1, dropout_prob)
            # output fully connected layer, neuron for each class
            oweight = tf.tile(tf.expand_dims(weights['dense_out'], axis=0), (tf.shape(x)[0], 1, 1))
            out_layer = tf.matmul(layer_1_drop, oweight) + biases['dense_out']

            cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_units) # Single layer LSTM
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
         				    inputs=out_layer,
           			    sequence_length=seqlen, # Tensorflow uses this to automatically mask out the zero padding
           			    dtype=tf.float32)

            last_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(outputs)[0]), seqlen-1], axis=1)) # This gets the last output before zero padding
            out = tf.matmul(last_output, weights['lstm_out']) + biases['lstm_out']

            return out

        logits = neural_net(inputs)
        prediction = tf.nn.softmax(logits) # reduce unscaled values to probabilities

        # Copied this shit from our other networks
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_labels)
        loss = tf.reduce_mean(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        train_op = optimizer.minimize(loss)

        # Eval Shit
        predictions = tf.argmax(prediction, 1)
        pred_scores = tf.reduce_max(prediction,1)
        correct_pred = tf.equal(predictions, tf.argmax(correct_labels, 1)) # check the index with the largest value
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

        evaluation_args = [loss, accuracy, predictions, pred_scores, correct_pred]

        return graph, inputs, correct_labels, seqlen, train_op, dropout_prob, evaluation_args


if __name__ == "__main__":
    # train the neural network on test data
    graph, X, Y, seqlen, optimizer, dropout_prob, evaluation_args = build_nn()
    train_device_cycle_nn(graph, X, Y, seqlen, optimizer, dropout_prob, evaluation_args, gen_data())
