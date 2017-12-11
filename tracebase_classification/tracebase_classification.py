#! /usr/bin/env python3

# Much of this was adapted from:
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

import tensorflow as tf
import numpy as np

# function to generate a training and validation, with equal label representation
def generate_training_and_validation (dataset, labelset, lengths, nameset, testing_percent):
    data_len = dataset.shape[1]
    if len(dataset.shape) > 2:
        channels = dataset.shape[2]
        training_data =     np.empty((0, data_len, channels))
        validation_data =   np.empty((0, data_len, channels))
    else:
        training_data =     np.empty((0, data_len))
        validation_data =   np.empty((0, data_len))
    training_labels =   np.empty((0, data_len))
    training_lengths =   np.empty((0, data_len))
    validation_labels = np.empty((0, data_len))
    validation_lengths =   np.empty((0, data_len))
    training_names =   np.empty((0, data_len))
    validation_names = np.empty((0, data_len))

    for label in sorted(range(int(max(labelset+1)))):
        # find indices matching this class
        matching_indices = np.flatnonzero((labelset == label))
        matching_data = dataset[matching_indices]
        matching_lengths = lengths[matching_indices]
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
            validation_lengths = np.append(validation_lengths, matching_lengths[device_matching_indices])
            validation_names = np.append(validation_names, matching_names[device_matching_indices])

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
            training_lengths= np.append(training_lengths, matching_lengths[device_matching_indices])
            training_names = np.append(training_names, matching_names[device_matching_indices])

    # shuffle the validation and data arrays
    permutation = np.random.permutation(training_data.shape[0])
    shuffled_training_data = training_data[permutation]
    shuffled_training_labels = training_labels[permutation]
    shuffled_training_lengths = training_lengths[permutation]
    shuffled_training_names = training_names[permutation]
    permutation = np.random.permutation(validation_data.shape[0])
    shuffled_validation_data = validation_data[permutation]
    shuffled_validation_labels = validation_labels[permutation]
    shuffled_validation_lengths = validation_lengths[permutation]
    shuffled_validation_names = validation_names[permutation]

    # complete! Return data
    return  shuffled_training_data, shuffled_training_labels,\
            shuffled_training_lengths, shuffled_training_names,\
            shuffled_validation_data, shuffled_validation_labels,\
            shuffled_validation_lengths, shuffled_validation_names

# function to create the training and validation datasets
def gen_data():
    # load and shuffle data
    Data = np.load("../tracebase_data/traces_bundle.npy")
    Labels = np.load("../tracebase_data/traces_classes.npy")
    Lengths = np.load("../tracebase_data/traces_lengths.npy")
    Names = np.load("../tracebase_data/traces_dev_ids.npy")
    num_names = np.max(Names) + 1

    # get label string names and pad spaces to make them equal length
    labelstrs = np.load("../tracebase_data/traces_class_map.npy")
    print(labelstrs)
    print(len(labelstrs))
    print(max(Labels)+1)
    print(Labels)
    print(Names)
    max_str_len = max([len(s) for s in labelstrs])
    for index, label in enumerate(labelstrs):
        labelstrs[index] = label + ' '*(max_str_len - len(label))

    # quick idiot test
    if max(Labels)+1 != len(labelstrs):
        print("Error: Number of classes doesn't match labels input")
        exit()

    # generate training and validation datasets (already shuffled)
    TrainingData, TrainingLabels, TrainingLengths, TrainingNames,\
    ValidationData, ValidationLabels, ValidationLengths, ValidationNames\
    = generate_training_and_validation(Data, Labels, Lengths, Names, 0.20)

    return (TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingLengths, ValidationLengths, TrainingNames, ValidationNames, labelstrs, num_names)


# Load data and generate train/validate
TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingLengths, ValidationLengths, TrainingNames, ValidationNames, labelstrs, num_names = gen_data()

# hyperparameters and config
n_hidden = 10
trace_len = len(TrainingData[0,:,0])
n_classes = np.argmax(np.unique(labelstrs)) + 1
learning_rate = 0.001
drop_probability = 0
batch_size = 25

x = tf.placeholder(tf.float32, (batch_size, trace_len, 2))
t = tf.placeholder(tf.float32, (batch_size, trace_len, 1))
y = tf.placeholder(tf.int32, (batch_size))
seqlen = tf.placeholder(tf.int32, (batch_size))

# define network
# LSTM layer + fully connected layer
inputs = (t, x)
outputs, _ = tf.nn.dynamic_rnn(cell= tf.contrib.rnn.PhasedLSTMCell(n_hidden), inputs=inputs, dtype=tf.float32, sequence_length = seqlen)
outputs_dropout = tf.nn.dropout(outputs, tf.constant(1.0 - drop_probability))
last_output = tf.gather_nd(outputs_dropout, tf.stack([tf.range(batch_size), seqlen-1], axis=1))

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [n_hidden, n_classes])
    b = tf.get_variable('b', [n_classes], initializer = tf.constant_initializer(0.0))
logits = tf.matmul(last_output, W) + b
print(logits.shape)
preds = tf.nn.softmax(logits)
correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#rnn_out = tf.squeeze(outputs[:, -1, :])
#y = slim.fully_connected(inputs= rnn_out, num_outputs= n_classes, activation_fn=tf.nn.tanh)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)


