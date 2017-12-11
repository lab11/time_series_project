#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import sys
import os
import argparse
import itertools
import time

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

# input directory
in_dir = '../plaid_data/barsim_et_al_data/'

# function to create the training and validation datasets
def gen_data ():
    # load and shuffle data
    data = np.load(in_dir + "traces_bundle.npy")
    np.random.shuffle(data)
    Data   = data[:, :-3]
    Labels = data[:, -1]
    Names  = data[:, -2]
    Houses = data[:, -3]
    num_names = np.max(Names) + 1

    # normalize all waveform magnitude to the maximum for that type
    data_len = len(Data[0])
    Data[:, :data_len//2] /= np.amax(np.absolute(Data[:, :data_len//2])) # current
    Data[:, data_len//2:] /= np.amax(np.absolute(Data[:, data_len//2:])) # voltage

    # get label string names and pad spaces to make them equal length
    labelstrs = np.load(in_dir + 'traces_class_map.npy')
    max_str_len = max([len(s) for s in labelstrs])
    for index, label in enumerate(labelstrs):
        labelstrs[index] = label + ' '*(max_str_len - len(label))

    namestrs = np.load(in_dir + 'traces_name_map.npy')

    # quick idiot test
    if max(Labels)+1 != len(labelstrs):
        print("Error: Number of classes doesn't match labels input")
        sys.exit()

    return (Data, Labels, Names, Houses, labelstrs, namestrs)

def get_input_len ():
    # length of data dimension, minus 3 (label, name, and house)
    return np.shape(np.load(in_dir + 'traces_bundle.npy'))[1] - 3

def get_labels_len ():
    # number of classes saved
    return np.shape(np.load(in_dir + 'traces_class_map.npy'))[0]

def train_neural_network (tf_inputs, tf_expecteds, train_ops, loss_ops, accuracy_ops, sess, \
        Data, Labels, class_count, \
        combo_index, class_a, class_b, combo_training_indices, combo_validation_indices):

    # configurations
    batch_size = 50
    display_step = 100

    # determine probabilities for each class
    class_a_count = np.sum(Labels[combo_training_indices] == class_a)
    class_a_each_prob = 0.5/class_a_count
    class_b_count = len(combo_training_indices)-class_a_count
    class_b_each_prob = 0.5/class_b_count
    combo_training_probabilities = np.where((Labels[combo_training_indices] == class_a), class_a_each_prob, class_b_each_prob)

    # convert labels into one-hot array
    ClassOneHotTrainingLabels = np.eye(class_count)[Labels[combo_training_indices].astype(np.int64)]
    OneHotComboTrainingLabels = np.transpose([ClassOneHotTrainingLabels[:, class_a], ClassOneHotTrainingLabels[:, class_b]])
    ClassOneHotValidationLabels = np.eye(class_count)[Labels[combo_validation_indices].astype(np.int64)]
    OneHotComboValidationLabels = np.transpose([ClassOneHotValidationLabels[:, class_a], ClassOneHotValidationLabels[:, class_b]])

    # select the correct neural network to run on
    tf_input = tf_inputs[combo_index]
    tf_expected = tf_expecteds[combo_index]
    train_op = train_ops[combo_index]
    loss_op = loss_ops[combo_index]
    accuracy_op = accuracy_ops[combo_index]

    # train the neural network on this data
    step = 0
    min_validation_loss = 5000
    while True:
        step += 1

        # select data to train on for this mini-batch
        batch_nums = np.random.choice(len(combo_training_indices), batch_size, p=combo_training_probabilities)

        # run training
        # warning: keep the indices as Data[combo[batch]], indexing the Data first via combo then via batch is 20x slower
        in_dat = Data[combo_training_indices[batch_nums]]
        in_lab = OneHotComboTrainingLabels[batch_nums]
        sess.run(train_op, feed_dict={tf_input: in_dat, tf_expected: in_lab})

        # check accuracy every N iterations
        if step % display_step == 0 or step == 1:

            # save the trainer
            if checkpointFile is not None:
                saver.save(sess, checkpointFile)

            # training accuracy
            training_loss, training_accuracy = sess.run([loss_op, accuracy_op], feed_dict={tf_input: Data[combo_training_indices], tf_expected: OneHotComboTrainingLabels})

            # validation accuracy
            validation_loss, validation_accuracy = sess.run([loss_op, accuracy_op], feed_dict={tf_input: Data[combo_validation_indices], tf_expected: OneHotComboValidationLabels})

            # early stopping generalization loss per "Early Stopping - But When?"
            #if validation_loss < min_validation_loss:
            #    min_validation_loss = validation_loss
            #generalization_loss = 100*(validation_loss/min_validation_loss - 1)
            generalization_loss = -1

            # print overall statistics
            print("\tStep {: >5d}, Training Loss= {: >8.3f}, Validation Loss= {: >8.3f}, Generalization Loss = {: >8.3f}".format(step, training_loss, validation_loss, generalization_loss))
            print("\t  Training Accuracy= {:.3f} Validation Accuracy= {:.3f}".format(training_accuracy, validation_accuracy))

            # check if we should stop
            # stop based on generalization loss increasing or a step limit
            validation_accuracy_limit = 0.999
            max_generalization_loss = 5.0 # picked as the highest GL they use in the paper
            desired_max_step = 30000
            step_limit = maxstep
            if step_limit == -1:
                step_limit = desired_max_step
            if step >= step_limit or generalization_loss > max_generalization_loss or validation_accuracy > validation_accuracy_limit:
                print("\tCompleted at step " + str(step))
                break

# function to run neural network training
def run_nn (tf_inputs, tf_expecteds, train_ops, loss_ops, accuracy_ops, prediction_ops, generated_data):
    Data, Labels, Names, Houses, labelstrs, namestrs = generated_data

    # record start time
    start_time = time.time()

    # create storage for per-house results
    house_count = int(max(Houses))+1
    class_count = len(labelstrs)
    house_statistics = np.zeros((house_count, 4)) # Correct count, Total Count, Grouped Correct Count, Grouped Total Count
    class_confusion_matrix = np.zeros((class_count, class_count))

    # iterate through class combinations, creating datasets
    class_list = []
    for class_a, class_b in itertools.combinations(range(class_count), 2):
        class_combo_indices = np.flatnonzero(np.any(((Labels == class_a), (Labels == class_b)), axis=0))
        class_list.append((class_a, class_b, class_combo_indices))

    # iterate through houses
    for house_index in range(house_count):
        print("Leave one out: House " + str(house_index+1))

        # determine training and validation sets
        training_indices = np.flatnonzero((Houses != house_index))
        validation_indices = np.flatnonzero((Houses == house_index))

        # initialize tensorflow variables
        init = tf.global_variables_initializer()

        # create a saver object
        saver = tf.train.Saver()

        # begin and initialize tensorflow session
        with tf.Session() as sess:
            sess.run(init)

            # set up saver
            if checkpointFile is not None:
                if os.path.isfile(checkpointFile + ".meta"):
                    print("Restoring model from " + checkpointFile)
                    saver.restore(sess, checkpointFile)

            # create pool of workers
            pool = mp.Pool()

            # iterate through class combinations
            for combo_index, (class_a, class_b, combo_indices) in enumerate(class_list):
                combo_training_indices = np.intersect1d(training_indices, combo_indices, assume_unique=True)
                combo_validation_indices = np.intersect1d(validation_indices, combo_indices, assume_unique=True)

                # check if the validation house has at least one of these devices
                if len(combo_validation_indices) == 0:
                    # it doesn't, so we can skip this entire NN
                    print(" Skipping   {:s} [{:s} {:s}]".format("House " + str(house_index+1), labelstrs[class_a], labelstrs[class_b]))
                    continue
                print(" Running    {:s} [{:s} {:s}]".format("House " + str(house_index+1), labelstrs[class_a], labelstrs[class_b]))

                pool.apply_async(train_neural_network, args = ( \
                        tf_inputs, tf_expecteds, train_ops, loss_ops, accuracy_ops, sess, \
                        Data, Labels, class_count, \
                        combo_index, class_a, class_b, combo_training_indices, combo_validation_indices) \
                        )

                # record results from the neural network
                print(" Complete")

            # wait until workers are done
            pool.close()
            pool.join()

            # calculate softmax predictions across all trained neural networks
            print(" Evaluating  {:s}".format("House " + str(house_index+1)))
            sum_of_softmaxes = np.zeros((len(validation_indices), class_count))
            for combo_index, (class_a, class_b, combo_indices) in enumerate(class_list):

                # select the correct neural network to run on
                tf_input = tf_inputs[combo_index]
                prediction_op = prediction_ops[combo_index]

                # run neural network on validation data
                softmaxes = sess.run(prediction_op, feed_dict={tf_input: Data[validation_indices]})

                # save results
                sum_of_softmaxes[:, class_a] += softmaxes[:, 0]
                sum_of_softmaxes[:, class_b] += softmaxes[:, 1]

            # determine correct predictions
            predictions = np.argmax(sum_of_softmaxes, axis=1)

            # per-house statistics
            correct_prediction_count = np.sum(np.equal(predictions, Labels[validation_indices].astype(int)))
            total_prediction_count = len(validation_indices)
            accuracy = correct_prediction_count / total_prediction_count
            print("\tAccuracy=  {:.5f}".format(accuracy))

            # calculate device trace-grouped voting
            correct_device_count = 0
            total_device_count = len(np.unique(Names[validation_indices]))
            for name in np.unique(Names[validation_indices]):
                device_name_indices = np.flatnonzero((Names[validation_indices] == name))
                prediction_sums = np.bincount(predictions[device_name_indices])
                device_prediction = np.argmax(prediction_sums)
                device_label = Labels[validation_indices[device_name_indices[0]]]
                if device_prediction == device_label:
                    correct_device_count += 1
            grouped_accuracy = correct_device_count/total_device_count
            print("\tGrouped Accuracy = {:.5f}".format(grouped_accuracy))

            # save house statistics
            house_statistics[house_index][0] = correct_prediction_count
            house_statistics[house_index][1] = total_prediction_count
            house_statistics[house_index][2] = correct_device_count
            house_statistics[house_index][3] = total_device_count
            np.save('house_statistics', house_statistics)

            # determine per-device statistics
            for index in range(len(validation_indices)):
                true_label = int(Labels[validation_indices[index]])
                predicted_label = int(predictions[index])
                class_confusion_matrix[true_label][predicted_label] += 1
            np.save('device_statistics', class_confusion_matrix)

            # done with this cross-validation set
            print("Complete")

            #XXX: figure out NN saving

    # wow, we actually finished?
    print("Done with everything!!! yay")
    print("That took " + str(start_time - time.time()) + " seconds")

