#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sys
import os
import argparse

# ensure that we always "randomly" run in a repeatable way
RANDOM_SEED = 21
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

#grab input arguments
parser = argparse.ArgumentParser(description='Run neural network')
parser.add_argument('-s', dest = "checkpointFile", type=str)
parser.add_argument('-n', dest = "maxstep", type=int, default=-1)
args = parser.parse_args()
checkpointFile = args.checkpointFile
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


# function to run neural network training
def run_nn(tf_input, tf_expected, train_op, loss_op, accuracy, predictions, pred_scores, correct_pred, generated_data):
    # configurations
    batch_size  = 50
    display_step  = 100

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

    # initialize tensorflow variables
    init = tf.global_variables_initializer()

    # create a saver object
    saver = tf.train.Saver()

    # begin and initialize tensorflow session
    with tf.Session() as sess:
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
            sess.run(train_op, feed_dict = {tf_input: TrainingData[batch_nums], tf_expected: OneHotTrainingLabels[batch_nums]})

            # check accuracy every N iterations
            if step % display_step == 0 or step == 1:

                #save the trainer
                if checkpointFile is not None:
                    saver.save(sess, checkpointFile)

                # training accuracy
                training_loss, training_accuracy, training_preds, training_pred_scores, training_correct_preds = sess.run([loss_op, accuracy, predictions,pred_scores, correct_pred], feed_dict={tf_input: TrainingData[training_nums], tf_expected: OneHotTrainingLabels[training_nums]})

                #calculate device grouped accuracy
                one_hot_preds = np.transpose(np.eye(len(labelstrs))[training_preds])
                one_hot_weighted_preds = np.dot(one_hot_preds,np.diag(training_pred_scores))
                ids = np.reshape(TrainingNames,[-1])
                one_hot_ids = np.eye(num_names.astype(int))[ids.astype(int)]
                votes = np.matmul(one_hot_preds, one_hot_ids)
                weighted_votes = np.matmul(one_hot_weighted_preds, one_hot_ids)
                votes = np.transpose(votes)
                weighted_votes = np.transpose(weighted_votes)
                good_votes = np.amax(votes,1)
                weighted_good_votes = np.amax(weighted_votes,1)
                not_included = np.not_equal(good_votes, 0)
                voted_labels = np.argmax(votes, 1)
                weighted_voted_labels = np.argmax(weighted_votes, 1)
                filtered_votes = voted_labels[not_included]
                filtered_weighted_votes = weighted_voted_labels[not_included]
                filtered_labels = id_to_labels[not_included]
                grouped_correct = filtered_votes == filtered_labels
                weighted_grouped_correct = filtered_weighted_votes == filtered_labels
                training_grouped_accuracy = np.mean(grouped_correct)
                training_grouped_weighted_accuracy = np.mean(weighted_grouped_correct)

                # validation accuracy
                validation_loss, validation_accuracy, validation_preds, validation_pred_scores, validation_correct_preds = sess.run([loss_op, accuracy, predictions, pred_scores, correct_pred], feed_dict={tf_input: ValidationData[validation_nums], tf_expected: OneHotValidationLabels[validation_nums]})

                #calculate device grouped accuracy
                one_hot_preds = np.transpose(np.eye(len(labelstrs))[validation_preds])
                one_hot_weighted_preds = np.dot(one_hot_preds,np.diag(validation_pred_scores))
                ids = np.reshape(ValidationNames,[-1])
                one_hot_ids = np.eye(num_names.astype(int))[ids.astype(int)]
                votes = np.matmul(one_hot_preds, one_hot_ids)
                weighted_votes = np.matmul(one_hot_weighted_preds, one_hot_ids)
                votes = np.transpose(votes)
                weighted_votes = np.transpose(weighted_votes)
                good_votes = np.amax(votes,1)
                weighted_good_votes = np.amax(weighted_votes,1)
                not_included = np.not_equal(good_votes, 0)
                voted_labels = np.argmax(votes, 1)
                weighted_voted_labels = np.argmax(weighted_votes, 1)
                filtered_votes = voted_labels[not_included]
                filtered_weighted_votes = weighted_voted_labels[not_included]
                filtered_labels = id_to_labels[not_included]
                grouped_correct = filtered_votes == filtered_labels
                weighted_grouped_correct = filtered_weighted_votes == filtered_labels
                validation_grouped_accuracy = np.mean(grouped_correct)
                validation_grouped_weighted_accuracy = np.mean(weighted_grouped_correct)

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

