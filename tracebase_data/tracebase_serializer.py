#! /usr/bin/env python3
import numpy as np
import csv
import glob
import os
import pathlib
import arrow
import matplotlib.pyplot as plt
from multiprocessing import Pool

"""
Takes all the tracebase files, turns them into numpy arrays, serializes them.
Csv files are [1-second avg power, 8-second avg power] in Watts.

How this thing works:
$ python3 plaid_serializer.py
"""

class_list = sorted([
    'Refridgerator',
    'MicrowaveOven',
    'Washingmachine',
    'VacuumCleaner',
    'PC-Laptop',
    'Printer',
    'Toaster',
    'Lamp',
    'Coffeemaker',
    'Cookingstove',
    'TV-LCD'])
print(class_list)

# Make folders if needed
if not os.path.exists('numpy_arrays'):
    os.makedirs('numpy_arrays')

# Get all the csv files names
fnames = glob.glob("tracebase/complete/*/*.csv")
files = {} # where we keep the csv files

# Iterate through files parsing and serializing as numpy arrays
print("Parsing files...")

def parse(fname):
    dev_class = fname.split('/')[-2]
    if dev_class not in class_list:
        return
    print('    ' + fname)
    a = np.loadtxt(fname, delimiter=";", dtype=str)
    converted = np.ndarray(a.shape, dtype = int)
    for i, measurement in enumerate(a):
        time = arrow.get(measurement[0],'DD/MM/YYYY HH:mm:ss')
        time_begin = time.floor('day')
        converted[i, 0]= (time - time_begin).seconds
    converted[:,1:] = a[:,1:].astype(int)

    # save numpy array
    basename = os.path.basename(fname).split('.')[0]
    firstword = basename.split('_')[0]
    if firstword == 'device':
        basename = basename.replace('device', 'dev')
    elif firstword != 'dev':
        if basename[0] == '_':
            basename = basename[1:]
        basename = 'dev_' + basename
    outfilename = "numpy_arrays/" +dev_class + '_' +basename
    np.save(outfilename, converted)

p = Pool(16)
p.map(parse,sorted(fnames))

#for fname in fnames:
#    parse(fname)
