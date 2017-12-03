#! /usr/bin/env python3

import numpy as np
import glob

fnames = glob.glob("plaid_paper_batch_data/*npy")

test = np.load(fnames[0])

# num traces x current | voltage | house | label
data = np.ndarray((len(fnames), test.shape[0]*test.shape[1]+2), dtype=float)

class_map = []

for fname, data_row in zip(sorted(fnames), data):
    class_name = fname.split('/')[1].split('-')[0]
    if class_name not in class_map:
        class_map.append(class_name)
    trace = np.load(fname)
    data_row[:-2] = np.reshape(trace, 2000)
    data_row[-2] = int(fname.split('-')[-2].split('house')[1])
    data_row[-1] = class_map.index(class_name)

# save big data array
np.save("traces_bundle", data)
np.save("traces_class_map", class_map)
