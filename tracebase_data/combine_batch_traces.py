#! /usr/bin/env python3

import numpy as np
import glob
import os
from zlib import crc32

fnames = glob.glob("numpy_arrays/*.npy")

test = np.load(fnames[0])

trace_array = []
class_map = []
name_map = []
labels = np.ndarray(len(fnames), dtype=int)
dev_ids= np.ndarray(len(fnames), dtype=int)
lengths = np.ndarray(len(fnames), dtype=int)

index = -1
for fname in sorted(fnames):
    # get unique device name from the filename
    # note: device name is a uniquely identifying string for that device including
    #   the dataset, house number, and model/make of the device
    device_name = os.path.basename(fname).split('_')[-2]

    # create map of device names we are including
    if device_name not in name_map:
        name_map.append(device_name)

    # get device class from the filename
    class_name = os.path.basename(fname).split('_')[0]
    if class_name not in class_map:
        class_map.append(class_name)

    # get the data row we should be filling with data
    index += 1
    #data_row = data[index]

    # record data for this trace
    trace = np.load(fname)
    trace_array.append(trace)

    #data_row[:-2, 0] = trace[:, 0]
    #data_row[:-2, 1] = trace[:, 1]
    labels[index] = class_map.index(class_name)
    dev_ids[index] = name_map.index(device_name)

min_len = 60*60*24
max_len = 0
for i, trace in enumerate(trace_array):
    length = len(trace)
    lengths[i] = length
    if length > max_len:
        max_len = length
    elif length < min_len:
        min_len = length

data = np.zeros((len(fnames), max_len, 3), dtype=float)
for i, data_row in enumerate(data):
    length = len(trace_array[i])
    data_row[:length] = trace_array[i][:]

# save big data array
dirname = "./"
np.save(dirname + "traces_bundle", data)
np.save(dirname + "traces_classes", labels)
np.save(dirname + "traces_dev_ids", dev_ids)
np.save(dirname + "traces_lengths", lengths)
np.save(dirname + "traces_class_map", class_map)
np.save(dirname + "traces_name_map", name_map)
