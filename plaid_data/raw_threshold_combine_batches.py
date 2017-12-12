#! /usr/bin/env python3

import numpy as np
import glob
import os
from zlib import crc32
import sys

fnames = glob.glob("raw_threshold_batch/numpy-power-event-*.npy")

data = np.zeros((len(fnames), 15000 + 2), dtype=float)
class_map = []
name_map = []

index = -1
for fname in sorted(fnames):
    # get unique device name from the filename
    # note: device name is a uniquely identifying string for that device including
    #   the dataset, house number, and model/make of the device
    device_name = os.path.basename(fname).split('-')[5]

    # create map of device names we are including
    if device_name not in name_map:
        name_map.append(device_name)

    # get device class from the filename
    class_name = os.path.basename(fname).split('-')[3]

    if class_name not in class_map:
        class_map.append(class_name)

    # get the data row we should be filling with data
    index += 1
    data_row = data[index]

    # record data for this trace
    trace = np.load(fname)
    data_row[:-2] = trace.flatten(order='F')
    data_row[-2] = name_map.index(device_name)
    data_row[-1] = class_map.index(class_name)

# save big data array
dirname = "./"
np.save(dirname + "raw_threshold_numpy_power_traces_bundle", data[0:index+1])
np.save(dirname + "raw_threshold_numpy_power_traces_class_map", class_map)
np.save(dirname + "raw_threshold_numpy_power_traces_name_map", name_map)
print("Done combining")
