#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
from plaid_data_setup import train_cycle_nn

import colored_traceback
colored_traceback.add_hook()

"""
Gonna test to see if feeding in half a second of an off-on transition to an LSTM results in good things
"""
RANDOM_SEED = 21

# function to create the training and validation datasets
def gen_data():
    # load and shuffle data
    data = np.load("../plaid_data/raw_threshold_numpy_power_traces_bundle.npy")
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(data)
    Data = data[:, 0:-2]
    Labels = data[:,-1]
    Names = data[:,-2]
    num_names = np.max(Names) + 1

    # normalize all waveform magnitude to the maximum for that type
    data_len = len(Data[0])
    Data[:, :data_len] /= np.amax(Data[:, :data_len]) # powerrrrrr
    Data = np.reshape(Data, (Data.shape[0],30, 500)) # Reshape data so that it holds individual cycles

    # get label string names and pad spaces to make them equal length
    labelstrs = np.load("../plaid_data/raw_threshold_numpy_power_traces_class_map.npy")
    max_str_len = max([len(s) for s in labelstrs])
    for index, label in enumerate(labelstrs):
        labelstrs[index] = label + ' '*(max_str_len - len(label))

    # quick idiot test
    if max(Labels)+1 != len(labelstrs):
        print("Error: Number of classes doesn't match labels input")
        sys.exit()

    # generate training and validation datasets (already shuffled)
    TrainingData, TrainingLabels, TrainingNames, ValidationData, ValidationLabels, ValidationNames = generate_training_and_validation(Data, Labels, Names, 0.20)

    return (TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names)

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

#Time to build our Trace LSTM
#test_sess = tf.InteractiveSession()

num_units = 400
num_labels = 11
learning_rate = .005

inputs = tf.placeholder(tf.float32, (None, 30, 500))
correct_labels = tf.placeholder(tf.float32, (None, num_labels))

cell = tf.contrib.rnn.BasicLSTMCell(num_units) # Single layer LSTM

outputs, state = tf.nn.dynamic_rnn(cell=cell,
								 inputs=inputs,
								 dtype=tf.float32)

last_output = tf.gather(outputs, int(outputs.get_shape()[1]) - 1, axis = 1)

# This variable_scope stuff seems unneccessary here, but seems like a good habit for tensorflow. Basically helps you manage variables when there are multiple neural networks
with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [num_units, num_labels])
	b = tf.get_variable('b', [num_labels], initializer = tf.constant_initializer(0.0))

logits = tf.matmul(last_output, W) + b
softmax = tf.nn.softmax(logits)

# Copied this shit from our other networks
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_labels)
loss = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_optimizer = optimizer.minimize(loss)

# Eval Shit
predictions = tf.argmax(softmax, 1)
pred_scores = tf.reduce_max(softmax,1)
correct_pred = tf.equal(predictions, tf.argmax(correct_labels, 1)) # check the index with the largest value
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

lstm_graph = tf.get_default_graph()
evaluation_args = [loss, accuracy, predictions, pred_scores, softmax, correct_pred]

train_cycle_nn(lstm_graph, inputs, correct_labels, train_optimizer, None, evaluation_args, gen_data())


#train_cycle_sequence_nn(lstm_graph, inputs, correct_labels, seqlen, train_op, evaluation_args, gen_data(), 'dense_single_layer')


"""

test_args = [inputs, outputs, last_output]

with tf.Session() as sess:
	sess.run(init)

	step = 0

	while True:
		step += 1



	test_in, test_puts, test_last_put = sess.run(test_args, feed_dict={inputs: synthetic_data, correct_labels: synthetic_correct, seqlen: synth_sequence_lengths})
	print(test_puts)
	print(tf.shape(test_puts))
	print(len(test_puts))
	print("--------------------------------")
	print(test_puts[0])
	print(tf.shape(test_puts[0]))
	print(len(test_puts[0]))
	print("+++++++++++++++++++++++++++++++")
	print(test_last_put)
	print(test_puts.shape)
	print(test_last_put.shape)
	#print(test_last_put)

	#print(test_last_put[0])
	#print(tf.shape(test_last_put[0]))
"""
