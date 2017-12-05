#! /usr/bin/env python3

import numpy as np
import glob
import os
from zlib import crc32

fnames = glob.glob("plaid_paper_batch_data/*.npy")

test = np.load(fnames[0])

# num traces x current | voltage | device_name | label
data = np.ndarray((len(fnames), test.shape[0]*test.shape[1]+2), dtype=float)

class_map = []
name_map = []

for fname, data_row in zip(sorted(fnames), data):
    # get device class from the filename
    class_name = os.path.basename(fname).split('-')[0]
    if class_name not in class_map:
        class_map.append(class_name)

    # get unique device name from the filename
    # note: device name is a uniquely identifying string for that device including
    #   the dataset, house number, and model/make of the device
    device_name = os.path.basename(fname).split('-')[2]
    if device_name not in name_map:
        name_map.append(device_name)

    trace = np.load(fname)
    data_row[:-2] = np.reshape(trace, 2000)
    data_row[-2] = name_map.index(device_name)
    data_row[-1] = class_map.index(class_name)

# save big data array
dirname = "./"
np.save(dirname + "traces_bundle", data)
np.save(dirname + "traces_class_map", class_map)
np.save(dirname + "traces_name_map", name_map)
