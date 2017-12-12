#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
import os
import argparse

# ensure that we always "randomly" run in a repeatable way
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#grab input arguments
parser = argparse.ArgumentParser(description='Run neural network')
parser.add_argument('-s', dest = "checkpointFile", type=str)
parser.add_argument('-l', dest = "LSTMcheckpointFile", type=str)
parser.add_argument('-n', dest = "maxstep", type=int, default=-1)
args = parser.parse_args()
checkpointFile = args.checkpointFile
LSTMcheckpointFile = args.LSTMcheckpointFile
maxstep = args.maxstep

# function to create the training and validation datasets
def gen_data():
    # load and shuffle data
    data = np.load("../plaid_data/traces_bundle.npy")
    np.random.shuffle(data)
    Data = data[:, 0:-2]
    Labels = data[:,-1]
    Names = data[:,-2]
    num_names = np.max(Names) + 1

    # normalize all waveform magnitude to the maximum for that type
    data_len = len(Data[0])
    Data[:, :data_len//2] /= np.amax(np.absolute(Data[:, :data_len//2])) # current
    Data[:, data_len//2:] /= np.amax(np.absolute(Data[:, data_len//2:])) # voltage

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
    TrainingData, TrainingLabels, TrainingNames, ValidationData, ValidationLabels, ValidationNames = generate_training_and_validation(Data, Labels, Names, 0.20)

    return (TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names)

def get_input_len():
    # length of data dimension, minus 2 (label and name)
    return np.shape(np.load("../plaid_data/traces_bundle.npy"))[1] - 2

def get_labels_len():
    # number of classes saved
    return np.shape(np.load("../plaid_data/traces_class_map.npy"))[0]


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

def group_accuracy_by_device(num_classes, num_devices, predictions, data_to_device_map, device_to_class_map):
    #calculate device grouped accuracy
    one_hot_preds = np.transpose(np.eye(num_classes)[predictions])
    ids = np.reshape(data_to_device_map,[-1])
    one_hot_ids = np.eye(num_devices)[ids.astype(int)]
    votes = np.matmul(one_hot_preds, one_hot_ids)
    votes = np.transpose(votes)
    good_votes = np.amax(votes,1)
    not_included = np.not_equal(good_votes, 0)
    voted_labels = np.argmax(votes, 1)
    filtered_votes = voted_labels[not_included]
    filtered_labels = device_to_class_map[not_included]
    grouped_correct = filtered_votes == filtered_labels
    return np.mean(grouped_correct)

def group_weighted_accuracy_by_device(num_classes, num_devices, predictions, prediction_scores, data_to_device_map, device_to_class_map):
    #calculate device grouped accuracy with class weights
    one_hot_preds = np.transpose(np.eye(num_classes)[predictions])
    one_hot_weighted_preds = np.dot(one_hot_preds,np.diag(prediction_scores))
    ids = np.reshape(data_to_device_map,[-1])
    one_hot_ids = np.eye(num_devices)[ids.astype(int)]
    weighted_votes = np.matmul(one_hot_weighted_preds, one_hot_ids)
    weighted_votes = np.transpose(weighted_votes)
    weighted_good_votes = np.amax(weighted_votes,1)
    not_included = np.not_equal(weighted_good_votes, 0)
    filtered_labels = device_to_class_map[not_included]
    weighted_voted_labels = np.argmax(weighted_votes, 1)
    filtered_weighted_votes = weighted_voted_labels[not_included]
    weighted_grouped_correct = filtered_weighted_votes == filtered_labels
    return np.mean(weighted_grouped_correct)

def train_cycle_hierarchy_nn(graph, tf_input, tf_expected, optimizer, evaluation_args, generated_data, cycle_classifier):
    #A cycle-sequence NN should take sequences (list which may be of dynamic size)
    #of the softmax output of a NN that attempts to classify raw cycles


    #by looking at the order and the distribution of these raw cycles it should
    #output the final class probability
    print("Training a sequence based NN...")
    print("Attempting to run a forward pass through the specified dense NN...")
    try:
        classifier = getattr(__import__(cycle_classifier), 'build_nn')
        gr, x, y, opt, drop, ev_args = classifier()
        print("Built network...")
        training_scores, training_names, training_labels, validation_scores, validation_names, validation_labels = run_cycle_nn(gr, x, y, ev_args, generated_data)
    except Exception as e:
        print(str(e))
        print("Failed to run forward pass on input data cycles. Exiting...")
        sys.exit(1)

    with graph.as_default():

        batch_size = 25
        dropout = 0.5
        display_step = 100

        #okay now that we have our data, we need to package it into score lists that can be fed into the higher layer network
        unique_training_devices = np.unique(training_names)
        training_device_data = np.zeros((len(unique_training_devices), len(np.unique(training_labels))))
        unique_validation_devices = np.unique(validation_names)
        validation_device_data = np.zeros((len(unique_validation_devices), len(np.unique(validation_labels))))
        training_device_label = np.zeros(len(unique_training_devices))
        validation_device_label = np.zeros(len(unique_validation_devices))

        for i in range(0, len(training_names)):
            training_device_data[np.where(unique_training_devices == training_names[i])[0][0]] += training_scores[i]
            training_device_label[np.where(unique_training_devices == training_names[i])[0][0]] = training_labels[i]

        for i in range(0, len(validation_names)):
            validation_device_data[np.where(unique_validation_devices == validation_names[i])[0][0]] += validation_scores[i]
            validation_device_label[np.where(unique_validation_devices == validation_names[i])[0][0]] = validation_labels[i]


        TrainingData = preprocessing.normalize(training_device_data, axis=1)
        ValidationData = preprocessing.normalize(validation_device_data, axis=1)

        TrainingLabels = np.array(training_device_label)
        ValidationLabels = np.array(validation_device_label)
        OneHotTrainingLabels = np.eye(len(np.unique(training_labels)))[TrainingLabels.astype(int)]
        OneHotValidationLabels = np.eye(len(np.unique(validation_labels)))[ValidationLabels.astype(int)]

        # initialize tensorflow variables
        init = tf.global_variables_initializer()

        # create a saver object
        saver = tf.train.Saver()

        # begin and initialize tensorflow session
        with tf.Session(graph=graph) as sess:
            sess.run(init)

            if LSTMcheckpointFile is not None:
                if os.path.isfile(LSTMcheckpointFile + ".meta"):
                    print("Restoring model from " + LSTMcheckpointFile)
                    saver.restore(sess, LSTMcheckpointFile)

            # iterate forever training model
            step = 1
            while True:
                step += 1

                # select data to train on and test on for this iteration
                batch_nums = np.random.choice(TrainingData.shape[0], batch_size)
                # run training
                sess.run(optimizer, feed_dict = {tf_input: TrainingData[batch_nums], tf_expected: OneHotTrainingLabels[batch_nums]})

                # check accuracy every N iterations
                if step % display_step == 0 or step == 1:

                    #save the trainer
                    if LSTMcheckpointFile is not None:
                        saver.save(sess, LSTMcheckpointFile)

                    # training accuracy
                    training_loss, training_accuracy, training_preds, training_pred_scores, training_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: TrainingData, tf_expected: OneHotTrainingLabels})

                    # validation accuracy
                    validation_loss, validation_accuracy, validation_preds, validation_pred_scores, validation_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: ValidationData, tf_expected: OneHotValidationLabels})

                    # print overal statistics
                    print("Step " + str(step) + \
                            ", Training Loss= " + "{: >8.3f}".format(training_loss) + \
                            ", Validation Loss= " + "{: >8.3f}".format(validation_loss))
                    print("--------------------------------------------------------------")

                    # determine per-class results and print
                    print("  Class                    | Training (cnt) | Validation (cnt)")
                    print("--------------------------------------------------------------")
                    #for label in range(len(labelstrs)):
                    #    label_indices = np.flatnonzero((TrainingLabels == label))
                    #    t_result = np.mean(training_correct_preds[label_indices])
                    #    t_count = len(label_indices)

                    #    label_indices = np.flatnonzero((ValidationLabels == label))
                    #    v_result = np.mean(validation_correct_preds[label_indices])
                    #    v_count = len(label_indices)

                    #    labelstr = labelstrs[label]

                    #    print("  {:s} |    {:.3f} ({:3d}) |      {:.3f} ({:3d})".format(labelstr, t_result, t_count, v_result, v_count))
                    #print("--------------------------------------------------------------")
                    print("  Total                    |    {:.3f}       |      {:.3f}".format(training_accuracy, validation_accuracy))
                    print(confusion_matrix(ValidationLabels, validation_preds))

                if maxstep != -1 and step >= maxstep:
                    print("Completed at step " + str(step))
                    sys.exit()


def train_cycle_sequence_nn(graph, tf_input, tf_expected, tf_sequences, optimizer, evaluation_args, generated_data, cycle_classifier):
    #A cycle-sequence NN should take sequences (list which may be of dynamic size)
    #of the softmax output of a NN that attempts to classify raw cycles
    # Expected structure is simple neural net into LSTM

    #by looking at the order and the distribution of these raw cycles it should
    #output the final class probability
    print("Training a sequence based NN...")
    print("Attempting to run a forward pass through the specified dense NN...")
    try:
        classifier = getattr(__import__(cycle_classifier), 'build_nn')
        gr, x, y, opt, drop, ev_args = classifier()
        print("Built network...")
        training_scores, training_names, training_labels, validation_scores, validation_names, validation_labels = run_cycle_nn(gr, x, y, ev_args, generated_data)
    except Exception as e:
        print(str(e))
        print("Failed to run forward pass on input data cycles. Exiting...")
        sys.exit(1)

    #okay now that we have our data, we need to package it into score lists that can be fed into the LSTM
    unique_training_devices = np.unique(training_names)
    training_device_data = [[] for i in range(len(unique_training_devices))]
    unique_validation_devices = np.unique(validation_names)
    validation_device_data = [[] for i in range(len(unique_validation_devices))]
    training_device_label = np.zeros(len(unique_training_devices))
    validation_device_label = np.zeros(len(unique_validation_devices))

    for i in range(0, len(training_names)):
        training_device_data[np.where(unique_training_devices == training_names[i])[0][0]].append(training_scores[i])
        training_device_label[np.where(unique_training_devices == training_names[i])[0][0]] = training_labels[i]

    for i in range(0, len(validation_names)):
        validation_device_data[np.where(unique_validation_devices == validation_names[i])[0][0]].append(validation_scores[i])
        validation_device_label[np.where(unique_validation_devices == validation_names[i])[0][0]] = validation_labels[i]

    #convert the data into a numpy array, zero pad the data, and generate the list of true sequences
    TrainingSequenceLengths = []
    max_len = 0
    for i in range(0, len(training_device_data)):
        TrainingSequenceLengths.append(len(training_device_data[i]))
        if(len(training_device_data[i]) > max_len):
            max_len = len(training_device_data[i])
    TrainingMaxLen = max_len
    TrainingSequenceLengths = np.array(TrainingSequenceLengths, dtype=np.int32)
    TrainingData = np.zeros((len(training_device_data), max_len, len(np.unique(training_labels))))

    # Do the same for the validation data
    ValidationSequenceLengths = []
    max_len = 0
    for i in range(0, len(validation_device_data)):
        ValidationSequenceLengths.append(len(validation_device_data[i]))
        if(len(validation_device_data[i]) > max_len):
            max_len = len(validation_device_data[i])
    ValidationMaxLen = max_len
    ValidationSequenceLengths = np.array(ValidationSequenceLengths, dtype=np.int32)
    ValidationData = np.zeros((len(validation_device_data), max_len, len(np.unique(training_labels))))

    for i in range(0, len(training_device_data)):
        for j in range(0, len(training_device_data[i])):
            TrainingData[i][j] = training_device_data[i][j]

    for i in range(0, len(validation_device_data)):
        for j in range(0, len(validation_device_data[i])):
            ValidationData[i][j] = validation_device_data[i][j]

    TrainingLabels = np.array(training_device_label)
    ValidationLabels = np.array(validation_device_label)
    OneHotTrainingLabels = np.eye(len(np.unique(training_labels)))[TrainingLabels.astype(int)]
    OneHotValidationLabels = np.eye(len(np.unique(validation_labels)))[ValidationLabels.astype(int)]

    # determine probabilities of selection for each trace
    #  - the probability of selecting each class should be equal
    #  - the probability of selecting any trace within a single class should be equal
    n_classes = len(np.unique(training_labels))
    class_prob = 1.0/n_classes
    device_prob = np.zeros(n_classes)
    TrainingData_probabilities = np.zeros(len(TrainingLabels))
    for label in TrainingLabels:
        device_prob[int(label)] += 1.0
    for index, count in enumerate(device_prob):
        device_prob[index] = class_prob/count
    for index in range(len(TrainingData_probabilities)):
        TrainingData_probabilities[index] = device_prob[int(TrainingLabels[index])]



    with graph.as_default():

        batch_size = 25
        dropout = 0.5
        display_step = 100

        # initialize tensorflow variables
        init = tf.global_variables_initializer()

        # create a saver object
        saver = tf.train.Saver()

        # begin and initialize tensorflow session
        with tf.Session(graph=graph) as sess:
            sess.run(init)

            if LSTMcheckpointFile is not None:
                if os.path.isfile(LSTMcheckpointFile + ".meta"):
                    print("Restoring model from " + LSTMcheckpointFile)
                    saver.restore(sess, LSTMcheckpointFile)


            # iterate forever training model
            step = 1
            while True:
                step += 1

                # select data to train on and test on for this iteration
                batch_nums = np.random.choice(TrainingData.shape[0], batch_size, p=TrainingData_probabilities)

                # run training
                sess.run(optimizer, feed_dict = {tf_input: TrainingData[batch_nums], tf_expected: OneHotTrainingLabels[batch_nums], tf_sequences: TrainingSequenceLengths[batch_nums]})

                # check accuracy every N iterations
                if step % display_step == 0 or step == 1:

                    #save the trainer
                    if LSTMcheckpointFile is not None:
                        saver.save(sess, LSTMcheckpointFile)

                    # training accuracy
                    training_loss, training_accuracy, training_preds, training_pred_scores, training_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: TrainingData, tf_expected: OneHotTrainingLabels, tf_sequences: TrainingSequenceLengths})

                    # validation accuracy
                    validation_loss, validation_accuracy, validation_preds, validation_pred_scores, validation_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: ValidationData, tf_expected: OneHotValidationLabels, tf_sequences: ValidationSequenceLengths})

                    # print overal statistics
                    print("Step " + str(step) + \
                            ", Training Loss= " + "{: >8.3f}".format(training_loss) + \
                            ", Validation Loss= " + "{: >8.3f}".format(validation_loss))
                    print("--------------------------------------------------------------")

                    # determine per-class results and print
                    print("  Class                    | Training (cnt) | Validation (cnt)")
                    print("--------------------------------------------------------------")
                    #for label in range(len(labelstrs)):
                    #    label_indices = np.flatnonzero((TrainingLabels == label))
                    #    t_result = np.mean(training_correct_preds[label_indices])
                    #    t_count = len(label_indices)

                    #    label_indices = np.flatnonzero((ValidationLabels == label))
                    #    v_result = np.mean(validation_correct_preds[label_indices])
                    #    v_count = len(label_indices)

                    #    labelstr = labelstrs[label]

                    #    print("  {:s} |    {:.3f} ({:3d}) |      {:.3f} ({:3d})".format(labelstr, t_result, t_count, v_result, v_count))
                    #print("--------------------------------------------------------------")
                    print("  Total                    |    {:.3f}       |      {:.3f}".format(training_accuracy, validation_accuracy))
                    print(confusion_matrix(ValidationLabels, validation_preds))

                if maxstep != -1 and step >= maxstep:
                    print("Completed at step " + str(step))
                    sys.exit()

# function to run neural network training
def train_cycle_nn(graph, tf_input, tf_expected, optimizer, dropout_prob, evaluation_args, generated_data):
    #a cycle NN takes in raw current/voltage waveforms for a cycle and attempts
    #to classify them

    # configurations
    batch_size  = 50
    display_step  = 100
    dropout = 0.5

    # create various test data
    TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names = generated_data

    # determine probabilities of selection for each trace
    #  - the probability of selecting each class should be equal
    #  - the probability of selecting any trace within a single class should be equal
    n_classes = len(labelstrs)
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

    # always test on everything
    training_nums = range(len(TrainingData))
    validation_nums = range(len(ValidationData))

    with graph.as_default():
        # initialize tensorflow variables
        init = tf.global_variables_initializer()

        # create a saver object
        saver = tf.train.Saver()

        # begin and initialize tensorflow session
        with tf.Session(graph=graph) as sess:
            sess.run(init)

            if checkpointFile is not None:
                if os.path.isfile(checkpointFile + ".meta"):
                    print("Restoring model from " + checkpointFile)
                    saver.restore(sess, checkpointFile)

            id_to_labels = np.zeros(num_names.astype(int))
            for i in range(0, num_names.astype(int)):
                index = np.where(TrainingNames == i)
                if(len(index[0]) > 0):
                    id_to_labels[i] = TrainingLabels[index[0][0]]

            for i in range(0, num_names.astype(int)):
                index = np.where(ValidationNames == i)
                if(len(index[0]) > 0):
                    id_to_labels[i] = ValidationLabels[index[0][0]]


            # iterate forever training model
            step = 1
            while True:
                step += 1

                # select data to train on and test on for this iteration
                batch_nums = np.random.choice(TrainingData.shape[0], batch_size, p=TrainingData_probabilities)

                # run training
                if(type(dropout_prob) != None):
                    sess.run(optimizer, feed_dict = {tf_input: TrainingData[batch_nums], tf_expected: OneHotTrainingLabels[batch_nums], dropout_prob: dropout})
                else:
                    sess.run(optimizer, feed_dict = {tf_input: TrainingData[batch_nums], tf_expected: OneHotTrainingLabels[batch_nums]})


                # check accuracy every N iterations
                if step % display_step == 0 or step == 1:

                    #save the trainer
                    if checkpointFile is not None:
                        saver.save(sess, checkpointFile)

                    # training accuracy
                    training_loss, training_accuracy, training_preds, training_pred_scores, training_pred_scores_full, training_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: TrainingData[training_nums], tf_expected: OneHotTrainingLabels[training_nums]})

                    training_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, TrainingNames, id_to_labels)
                    training_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, training_pred_scores, TrainingNames, id_to_labels)

                    # validation accuracy
                    validation_loss, validation_accuracy, validation_preds, validation_pred_scores, validation_pred_scores_full, validation_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: ValidationData[validation_nums], tf_expected: OneHotValidationLabels[validation_nums]})

                    validation_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, ValidationNames, id_to_labels)
                    validation_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, validation_pred_scores, ValidationNames, id_to_labels)

                    # print overal statistics
                    print("Step " + str(step) + \
                            ", Training Loss= " + "{: >8.3f}".format(training_loss) + \
                            ", Validation Loss= " + "{: >8.3f}".format(validation_loss))
                    print("--------------------------------------------------------------")

                    # determine per-class results and print
                    print("  Class                    | Training (cnt) | Validation (cnt)")
                    print("--------------------------------------------------------------")
                    for label in range(len(labelstrs)):
                        label_indices = np.flatnonzero((TrainingLabels == label))
                        t_result = np.mean(training_correct_preds[label_indices])
                        t_count = len(label_indices)

                        label_indices = np.flatnonzero((ValidationLabels == label))
                        v_result = np.mean(validation_correct_preds[label_indices])
                        v_count = len(label_indices)

                        labelstr = labelstrs[label]

                        print("  {:s} |    {:.3f} ({:3d}) |      {:.3f} ({:3d})".format(labelstr, t_result, t_count, v_result, v_count))
                    print("--------------------------------------------------------------")
                    print("  Total                    |    {:.3f}       |      {:.3f}".format(training_accuracy, validation_accuracy))
                    print("  Grouped Total            |    {:.3f}       |      {:.3f}".format(training_grouped_accuracy, validation_grouped_accuracy))
                    print("  Weighted Grouped Total   |    {:.3f}       |      {:.3f}".format(training_grouped_weighted_accuracy, validation_grouped_weighted_accuracy))
                    print(confusion_matrix(ValidationLabels[validation_nums], validation_preds))

                if maxstep != -1 and step >= maxstep:
                    print("Completed at step " + str(step))
                    sys.exit()

def run_cycle_nn(graph, tf_input, tf_expected, evaluation_args, generated_data):
    #runs a forward pass on the cycle NN

    # create various test data
    TrainingData, ValidationData, TrainingLabels, ValidationLabels, TrainingNames, ValidationNames, labelstrs, num_names = generated_data


    # convert Labels from integers to one-hot array
    n_classes = len(labelstrs)
    OneHotTrainingLabels = np.eye(n_classes)[TrainingLabels.astype(np.int64)]
    OneHotValidationLabels = np.eye(n_classes)[ValidationLabels.astype(np.int64)]

    # initialize tensorflow variables
    with graph.as_default():

        init = tf.global_variables_initializer()
        print(tf.get_default_graph())

        saver = tf.train.Saver()

        # begin and initialize tensorflow session
        with tf.Session(graph=graph) as sess:
            # create a saver object
            sess.run(init)

            if checkpointFile is not None:
                if os.path.isfile(checkpointFile + ".meta"):
                    print("Restoring model from " + checkpointFile)
                    saver.restore(sess, checkpointFile)
                else:
                    print("Network restore failed. Cannot run a forward pass with no network to restore from. Exiting.")
                    sys.exit(1)
            else:
                print("Must pass restore file. Cannot run a forward pass with no network to restore from. Exiting.")
                sys.exit(1)

            id_to_labels = np.zeros(num_names.astype(int))
            for i in range(0, num_names.astype(int)):
                index = np.where(TrainingNames == i)
                if(len(index[0]) > 0):
                    id_to_labels[i] = TrainingLabels[index[0][0]]

            for i in range(0, num_names.astype(int)):
                index = np.where(ValidationNames == i)
                if(len(index[0]) > 0):
                    id_to_labels[i] = ValidationLabels[index[0][0]]

            # training accuracy
            training_loss, training_accuracy, training_preds, training_pred_scores, training_pred_scores_full, training_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: TrainingData, tf_expected: OneHotTrainingLabels})

            training_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, TrainingNames, id_to_labels)
            training_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), training_preds, training_pred_scores, TrainingNames, id_to_labels)

            # validation accuracy
            validation_loss, validation_accuracy, validation_preds, validation_pred_scores, validation_pred_scores_full, validation_correct_preds = sess.run(evaluation_args, feed_dict={tf_input: ValidationData, tf_expected: OneHotValidationLabels})

            validation_grouped_accuracy = group_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, ValidationNames, id_to_labels)
            validation_grouped_weighted_accuracy = group_weighted_accuracy_by_device(len(labelstrs), num_names.astype(int), validation_preds, validation_pred_scores, ValidationNames, id_to_labels)

            print("")
            print("")
            print("Training Loss= " + "{: >8.3f}".format(training_loss) + \
                    ", Validation Loss= " + "{: >8.3f}".format(validation_loss))
            print("--------------------------------------------------------------")

            # determine per-class results and print
            print("  Class                    | Training (cnt) | Validation (cnt)")
            print("--------------------------------------------------------------")
            for label in range(len(labelstrs)):
                label_indices = np.flatnonzero((TrainingLabels == label))
                t_result = np.mean(training_correct_preds[label_indices])
                t_count = len(label_indices)

                label_indices = np.flatnonzero((ValidationLabels == label))
                v_result = np.mean(validation_correct_preds[label_indices])
                v_count = len(label_indices)

                labelstr = labelstrs[label]

                print("  {:s} |    {:.3f} ({:3d}) |      {:.3f} ({:3d})".format(labelstr, t_result, t_count, v_result, v_count))
            print("--------------------------------------------------------------")
            print("  Total                    |    {:.3f}       |      {:.3f}".format(training_accuracy, validation_accuracy))
            print("  Grouped Total            |    {:.3f}       |      {:.3f}".format(training_grouped_accuracy, validation_grouped_accuracy))
            print("  Weighted Grouped Total   |    {:.3f}       |      {:.3f}".format(training_grouped_weighted_accuracy, validation_grouped_weighted_accuracy))
            print(confusion_matrix(ValidationLabels, validation_preds))
            print("")
            print("")

            return training_pred_scores_full, TrainingNames, TrainingLabels, validation_pred_scores_full, ValidationNames, ValidationLabels
