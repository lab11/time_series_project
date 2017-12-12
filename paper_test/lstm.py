#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
import sys

from plaid_data_setup import get_input_len, get_labels_len, train_cycle_nn, train_device_cycle_nn, gen_data, gen_all_data


def build_nn():

    graph = tf.Graph()
    with graph.as_default():

        # Config:
        n_hidden         = 30*11
        n_dense_out      = 20
        n_lstm_units     = 50
        n_input          = get_input_len()
        n_labels         = get_labels_len()
        learning_rate    = 0.001

        inputs = tf.placeholder(tf.float32, (None, 10, n_input))
        correct_labels = tf.placeholder(tf.float32, (None, n_labels))
        # The true sequence lengths of the inputs. The actual inputs
        # should be zero-padded such that they are all the same length.
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
            cell = tf.contrib.rnn.BasicLSTMCell(n_lstm_units) # Single layer LSTM
            outputs, state = tf.nn.dynamic_rnn(cell=cell,
         				    inputs=x,
           			            dtype=tf.float32)

            last_output = tf.gather(outputs, int(outputs.get_shape()[1]) - 1, axis = 1)
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
        pred_scores_full = prediction
        correct_pred = tf.equal(predictions, tf.argmax(correct_labels, 1)) # check the index with the largest value
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

        evaluation_args = [loss, accuracy, predictions, pred_scores, pred_scores_full, correct_pred]

        return graph, inputs, correct_labels, train_op, dropout_prob, evaluation_args


if __name__ == "__main__":
    # train the neural network on test data
    graph, X, Y, optimizer, dropout_prob, evaluation_args = build_nn()
    train_cycle_nn(graph, X, Y, optimizer, dropout_prob, evaluation_args, gen_all_data())
