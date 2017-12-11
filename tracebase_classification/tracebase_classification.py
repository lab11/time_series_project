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
TrainingTimes = np.expand_dims(TrainingData[:,:,0], -1)
ValidationTimes = np.expand_dims(ValidationData[:,:,0], -1)
TrainingData = TrainingData[:,:,1:]
ValidationData = ValidationData[:,:,1:]

# hyperparameters and config
n_hidden = 10
trace_len = len(TrainingData[0,:,0])
n_classes = np.argmax(np.unique(labelstrs)) + 1
learning_rate = 0.001
drop_probability = 0
batch_size = 5
display_step  = 1

x = tf.placeholder(tf.float32, (None, trace_len, 2))
t = tf.placeholder(tf.float32, (None, trace_len, 1))
y = tf.placeholder(tf.float32, (None))
seqlen = tf.placeholder(tf.float32, (None))

# define network
# LSTM layer + fully connected layer
inputs = (t, x)
cell = tf.contrib.rnn.PhasedLSTMCell(n_hidden)
#initial_state = cell.zero_state(x.shape[0], tf.float32)
outputs, _ = tf.nn.dynamic_rnn(cell= cell, inputs=inputs, dtype=tf.float32, sequence_length = seqlen)
outputs_dropout = tf.nn.dropout(outputs, tf.constant(1.0 - drop_probability))
last_output = tf.gather_nd(outputs_dropout, tf.stack([tf.range(tf.shape(outputs_dropout)[0]), tf.cast(seqlen-1, tf.int32)], axis=1))

with tf.variable_scope('softmax'):
    W = tf.get_variable('W', [n_hidden, n_classes])
    b = tf.get_variable('b', [n_classes], initializer = tf.constant_initializer(0.0))
logits = tf.matmul(last_output, W) + b
preds = tf.nn.softmax(logits)
correct = tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), tf.cast(y, tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

#rnn_out = tf.squeeze(outputs[:, -1, :])
#y = slim.fully_connected(inputs= rnn_out, num_outputs= n_classes, activation_fn=tf.nn.tanh)

loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y, tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
evaluation_args = [loss_op, accuracy, preds]

# always test on everything
training_nums = range(len(TrainingData))
validation_nums = range(len(ValidationData))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    step = 0
    while True:
        step += 1


        # select data to train on and test on for this iteration
        batch_nums = np.random.choice(TrainingData.shape[0], batch_size)

        # run training
        sess.run(train_op, feed_dict = {x: TrainingData[batch_nums], t: TrainingTimes[batch_nums], y: TrainingLabels[batch_nums], seqlen: TrainingLengths[batch_nums]})

        # check accuracy every N iterations
        if step % display_step == 0 or step == 1:

            # training accuracy
            training_loss, training_accuracy, training_preds = sess.run(evaluation_args, feed_dict={x: TrainingData[training_nums], t: TrainingTimes[training_nums], y: TrainingLabels[training_nums], seqlen: TrainingLengths[training_nums]})

            #training_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, TrainingNames, id_to_labels)
            #training_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, training_pred_scores, TrainingNames, id_to_labels)

            # validation accuracy
            validation_loss, validation_accuracy, validation_preds = sess.run(evaluation_args, feed_dict={x: ValidationData[validation_nums], t: ValidationTimes[validation_nums], y: ValidationLabels[validation_nums], seqlen: ValidationLengths[validation_nums]})

            #validation_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, ValidationNames, id_to_labels)
            #validation_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, validation_pred_scores, ValidationNames, id_to_labels)

            # print overal statistics
            print("Step " + str(step) + \
                    ", Training Loss= " + "{: >8.3f}".format(training_loss) + \
                    ", Training Accuracy= " + "{: >8.3f}".format(training_accuracy) + \
                    ", Validation Loss= " + "{: >8.3f}".format(validation_loss) + \
                    ", Validation Accuracy= " + "{: >8.3f}".format(validation_accuracy))
