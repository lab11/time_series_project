#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
import sys

from plaid_data_setup import get_input_len, get_labels_len, train_cycle_nn, gen_cross_validation_data, select_cross_validation_set


def build_nn():

    graph = tf.Graph()
    with graph.as_default():

        # Config:
        n_hidden         = 30*11
        n_input          = get_input_len()
        n_labels         = get_labels_len()
        learning_rate    = 0.001

        # neural network inputs and expected results
        X = tf.placeholder("float", [None, n_input])
        Y = tf.placeholder("float", [None, n_labels])
        dropout_prob = tf.placeholder_with_default(1.0, shape=())

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
            layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
            # dropout on hidden layer
            layer_1_drop = tf.nn.dropout(layer_1, dropout_prob)
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
        pred_scores = tf.reduce_max(prediction,1)
        pred_scores_full = prediction
        correct_pred = tf.equal(predictions, tf.argmax(Y, 1)) # check the index with the largest value
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

        return graph, X, Y, train_op, dropout_prob, [loss_op, accuracy, predictions, pred_scores, pred_scores_full, correct_pred]

if __name__ == "__main__":

    # create dataset
    cross_validation_set_count = 5
    Data, Labels, Names, labelstrs, num_names, cross_validation_indices = gen_cross_validation_data(cross_validation_set_count)

    # run cross validation
    accuracies = []
    for cross_validation_index in range(cross_validation_set_count):

        # select cross validation set
        training_indices, validation_indices = select_cross_validation_set(cross_validation_indices, cross_validation_index)
        training_data = Data[training_indices]
        training_labels = Labels[training_indices]
        training_names = Names[training_indices]
        validation_data = Data[validation_indices]
        validation_labels = Labels[validation_indices]
        validation_names = Names[validation_indices]
        data_input = (training_data, validation_data, training_labels, validation_labels, training_names, validation_names, labelstrs, num_names)

        # train the neural network on test data
        graph, X, Y, optimizer, dropout_prob, evaluation_args = build_nn()
        accuracy = train_cycle_nn(graph, X, Y, optimizer, dropout_prob, evaluation_args, data_input)
        accuracies.append(accuracy)

    # print results
    print()
    print("Accuracies:", accuracies)
    print("Average accuracy=", np.mean(accuracies))

