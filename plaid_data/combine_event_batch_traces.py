#! /usr/bin/env python3

import numpy as np
import glob
import os
from zlib import crc32

fnames = glob.glob("plaid_paper_batch_data_events/*.npy")

# we're going to skip house2 from dataset2 as it has one of every class
# that dataset will be used for final validation
skiphouse = 'house2'
skipdataset = 'dataset2'
print("Skipping " + skiphouse + " from " + skipdataset)

class_map = []
name_map = []

max_cycles = 0
for fname in sorted(fnames):
    trace = np.load(fname)
    if trace.shape[0] > max_cycles:
        max_cycles = trace.shape[0]

print(max_cycles)

data = np.zeros((len(fnames), max_cycles, 500, 2))
nameclass = np.zeros((len(fnames), 2))

index = -1
for fname in sorted(fnames):
    # get unique device name from the filename
    # note: device name is a uniquely identifying string for that device including
    #   the dataset, house number, and model/make of the device
    device_name = os.path.basename(fname).split('-')[3]

    # skip validation house
    if skiphouse in device_name and skipdataset in device_name:
        continue

    # create map of device names we are including
    if device_name not in name_map:
        name_map.append(device_name)

    # get device class from the filename
    class_name = os.path.basename(fname).split('-')[1]
    if class_name not in class_map:
        class_map.append(class_name)

    # get the data row we should be filling with data
    index += 1
    data_row = data[index]
    nameclass_row = nameclass[index]

    # record data for this trace
    trace = np.load(fname)
    data_row = trace
    nameclass_row[0] = name_map.index(device_name)
    nameclass_row[1] = class_map.index(class_name)

# save big data array
dirname = "./"
np.save(dirname + "traces_event_bundle", data[0:index+1])
np.save(dirname + "traces_event_nameclass", nameclass[0:index+1])
np.save(dirname + "traces_event_class_map", class_map)
np.save(dirname + "traces_event_name_map", name_map)
