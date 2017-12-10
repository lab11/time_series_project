#! /usr/bin/env python3

import tensorflow as tf
import numpy as np

# function to generate a training and validation, with equal label representation
def generate_training_and_validation (dataset, labelset, nameset, testing_percent):
    data_len = dataset.shape[1]
    if len(dataset.shape) > 2:
        channels = dataset.shape[2]
        training_data =     np.empty((0, data_len, channels))
        validation_data =   np.empty((0, data_len, channels))
    else:
        training_data =     np.empty((0, data_len))
        validation_data =   np.empty((0, data_len))
    training_labels =   np.empty((0, data_len))
    validation_labels = np.empty((0, data_len))
    training_names =   np.empty((0, data_len))
    validation_names = np.empty((0, data_len))

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
            training_names = np.append(training_names, matching_names[device_matching_indices])

    # shuffle the validation and data arrays
    permutation = np.random.permutation(training_data.shape[0])
    shuffled_training_data = training_data[permutation]
    shuffled_training_labels = training_labels[permutation]
    shuffled_training_names = training_names[permutation]
    permutation = np.random.permutation(validation_data.shape[0])
    shuffled_validation_data = validation_data[permutation]
    shuffled_validation_labels = validation_labels[permutation]
    shuffled_validation_names = validation_names[permutation]

    # complete! Return data
    return shuffled_training_data, shuffled_training_labels, shuffled_training_names, shuffled_validation_data, shuffled_validation_labels, shuffled_validation_names

# function to create the training and validation datasets
def gen_data():
    # load and shuffle data
    Data = np.load("../tracebase_data/traces_bundle.npy")
    Labels = np.load("../tracebase_data/traces_classes.npy")
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
    TrainingData, TrainingLabels, TrainingNames, ValidationData, ValidationLabels, ValidationNames = generate_training_and_validation(Data, Labels, Names, 0.20)

    return (TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names)


def get_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

def cost(output, target):
    # cross entropy for each relevant frame
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    # Average over sequence lengths
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

# Load data and generate train/validate
TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names = gen_data()

# hyperparameters and config
n_hidden = 10
trace_len = len(TrainingData[0,:,0])
n_classes = np.argmax(np.unique(TrainingLabels)) + 1
learning_rate = 0.001
drop_probability = 0
batch_size = 25

X = tf.placeholder(tf.float32, (None, trace_len, 2))
T = tf.placeholder(tf.float32, (None, trace_len, 1))
Y = tf.placeholder(tf.float32, (None, n_classes))
sequence = tf.placeholder(tf.float32, [None, trace_len, 2])

# define network
# LSTM layer + fully connected layer
inputs = (T, X)
seqlen = get_length(sequence)
outputs, _ = tf.nn.dynamic_rnn(cell= tf.contrib.rnn.PhasedLSTMCell(n_hidden), inputs=inputs, dtype=tf.float32, sequence_length = seqlen)
outputs_dropout = tf.nn.dropout(outputs, tf.constant(1.0 - drop_probability))
last_output = tf.gather_nd(outputs_dropout, tf.stack([tf.range(batch_size), seqlen-1], axis=1))
exit()
# Everything below here is unfinished
#
#rnn_out = tf.squeeze(outputs[:, -1, :])
#y = slim.fully_connected(inputs= rnn_out, num_outputs= n_classes, activation_fn=tf.nn.tanh)

loss_op = cost(y, Y)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)


