#! /usr/bin/env python3

import numpy as np
import sys
import os
import glob

house_stats_name = 'house_statistics'
device_stats_name = 'device_statistics'

# go through saved stats files
fnames = glob.glob(house_stats_name + '*.npy')
for house_stats_file in fnames:
    device_stats_file = device_stats_name + house_stats_file.split(house_stats_name)[-1]

    print("Files: {:s} and {:s}".format(house_stats_file, device_stats_file))

    house_stats = np.load(house_stats_file)
    class_confusion_matrix = np.load(device_stats_file)

    print("Accuracy on validation sets:")
    for index, data_row in enumerate(house_stats):
        correct_count, total_count, grouped_correct_count, grouped_total_count = data_row
        print("    House {: >2d}: Accuracy= {:.5f}  Grouped Accuracy= {:.5f}".format(index+1, correct_count/total_count, grouped_correct_count/grouped_total_count))

    avg_accuracy = np.mean(house_stats[:, 0] / house_stats[:,1])
    avg_grouped_accuracy = np.mean(house_stats[:, 2] / house_stats[:,3])
    print("Total (mean): Accuracy= {:.5f}  Grouped Accuracy= {:.5f}".format(avg_accuracy, avg_grouped_accuracy))
    print()

    print("Overall device confusion matrix:")
    print(class_confusion_matrix.astype(int))
    print('\n')

