#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import sys

# Config:
n_hidden    = 30*11
n_input     = 2*2*500 # current & voltage, 2 AC cycles @ 30 kHz
n_classes   = 11
learning_rate = 0.001
display_step= 100
batch_size  = 50

# ensure that we always "randomly" run in a repeatable way
RANDOM_SEED = 21
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# load and shuffle data
data = np.load("../plaid_data/traces_bundle.npy")
np.random.shuffle(data)
Data = data[:, 0:-2]
Labels = data[:,-1]
if Labels.max()+1 != n_classes:
    print("Error: Number of classes doesn't match labels input")
    sys.exit()
Names = data[:,-2]

# function to generate a training and validation, with equal label representation
def generate_training_and_validation (dataset, labelset, nameset, testing_percent):
    training_data = np.empty((0,2000))
    training_labels = np.empty((0, 2000))
    validation_data = np.empty((0, 2000))
    validation_labels = np.empty((0, 2000))

    for label in sorted(range(int(max(labelset+1)))):
        # find indices matching this class
        matching_indices = np.flatnonzero((labelset == label))
        matching_data = dataset[matching_indices]
        matching_labels = labelset[matching_indices]
        matching_names = nameset[matching_indices]

        # determine number of traces we need to leave out
        class_trace_count = len(matching_names)
        min_trace_count = int(testing_percent*class_trace_count)

        # iterate through each device in this class
        unique_names, unique_name_counts = np.unique(matching_names, return_counts=True)
        selected_count = 0
        while selected_count < min_trace_count:
            # select an index
            index = np.random.randint(0, len(unique_names))

            # can't select the same device twice
            if unique_name_counts[index] == -1:
                continue

            # add selected device to validation set
            name = unique_names[index]
            device_matching_indices = np.flatnonzero((matching_names == name))
            validation_data = np.vstack((validation_data, matching_data[device_matching_indices]))
            validation_labels = np.append(validation_labels, matching_labels[device_matching_indices])

            # add trace count from selected device
            selected_count += unique_name_counts[index]
            unique_name_counts[index] = -1

        # add the rest to the training data
        for index in range(len(unique_names)):
            # skip validation data
            if unique_name_counts[index] == -1:
                continue

            # add to training set
            name = unique_names[index]
            device_matching_indices = np.flatnonzero((matching_names == name))
            training_data = np.vstack((training_data, matching_data[device_matching_indices]))
            training_labels = np.append(training_labels, matching_labels[device_matching_indices])


    # shuffle the validation and data arrays
    permutation = np.random.permutation(training_data.shape[0])
    shuffled_training_data = training_data[permutation]
    shuffled_training_labels = training_labels[permutation]
    permutation = np.random.permutation(validation_data.shape[0])
    shuffled_validation_data = validation_data[permutation]
    shuffled_validation_labels = validation_labels[permutation]

    # complete! Return data
    return shuffled_training_data, shuffled_training_labels, shuffled_validation_data, shuffled_validation_labels


# generate training and validation datasets (already shuffled)
TrainingData, TrainingLabels, ValidationData, ValidationLabels = generate_training_and_validation(Data, Labels, Names, 0.20)

# determine probabilities of selection for each trace
#  - the probability of selecting each class should be equal
#  - the probability of selecting any trace within a single class should be equal
class_prob = 1.0/n_classes
trace_prob = np.zeros(n_classes)
TrainingData_probabilities = np.zeros(len(TrainingLabels))
for label in TrainingLabels:
    trace_prob[int(label)] += 1.0
for index, count in enumerate(trace_prob):
    trace_prob[index] = class_prob/count
for index in range(len(TrainingData_probabilities)):
    TrainingData_probabilities[index] = trace_prob[int(TrainingLabels[index])]

# convert Labels from integers to one-hot array
OneHotTrainingLabels = np.eye(n_classes)[TrainingLabels.astype(np.int64)]
OneHotValidationLabels = np.eye(n_classes)[ValidationLabels.astype(np.int64)]

# neural network inputs
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# neural network parameters
weights = {
    'h1':  tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
}
biases = {
    'b1':   tf.Variable(tf.random_normal([n_hidden])),
    'out':  tf.Variable(tf.random_normal([n_classes])),
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
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1)) # check the index with the largest value
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    # always test on everything
    training_nums = range(len(TrainingData))
    validation_nums = range(len(ValidationData))

    # iterate forever training model
    step = 1
    while True:
        step += 1

        # select data to train on and test on for this iteration
        batch_nums = np.random.choice(TrainingData.shape[0], batch_size, p=TrainingData_probabilities)

        # run training
        sess.run(train_op, feed_dict = {X: TrainingData[batch_nums], Y: OneHotTrainingLabels[batch_nums]})

        # check accuracy every N iterations
        if step % display_step == 0 or step == 1:

            # training accuracy
            training_loss, training_accuracy, training_preds = sess.run([loss_op, accuracy, correct_pred], feed_dict={X: TrainingData[training_nums], Y:OneHotTrainingLabels[training_nums]})

            # validation accuracy
            validation_loss, validation_accuracy, validation_preds = sess.run([loss_op, accuracy, correct_pred], feed_dict={X: ValidationData[validation_nums], Y: OneHotValidationLabels[validation_nums]})

            # print overal statistics
            print("Step " + str(step) + \
                    ", Training Loss= " + "{:4.3f}".format(training_loss) + \
                    ", Validation Loss= " + "{:4.3f}".format(validation_loss))

            # determine per-class results and print
            print("Class | Training (cnt) | Validation (cnt)")
            for label in range(n_classes):
                label_indices = np.flatnonzero((TrainingLabels == label))
                t_result = np.mean(training_preds[label_indices])
                t_count = len(label_indices)

                label_indices = np.flatnonzero((ValidationLabels == label))
                v_result = np.mean(validation_preds[label_indices])
                v_count = len(label_indices)

                print("  {:3d} |    {:.3f} ({:3d}) |      {:.3f} ({:3d})".format(label, t_result, t_count, v_result, v_count))
            print("Total |    {:.3f}       |      {:.3f}".format(training_accuracy, validation_accuracy))

