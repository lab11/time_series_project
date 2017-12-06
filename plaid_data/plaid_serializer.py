#! /usr/bin/env python3
import numpy as np
import csv
import glob
import os
import pathlib
from multiprocessing import Pool

"""
Takes all the plaid files, turns them into numpy arrays, serializes them.
Csv files are [Current, Voltage] in Amps and Volts. Kind of inferring the Amps part based on numbers and graphs in papers
Sample time and frequency are stored in meta.json files

How this thing works:
$ python3 plaid_serializer.py

Author: Will Huang
"""

# get plaid dataset if necessary
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    os.system("wget http://plaidplug.com/static/dataset/Plaid.tar.gz")
    os.system("tar -xvf Plaid.tar.gz")

# Make folders if needed
if not os.path.exists('numpy_arrays'):
    os.makedirs('numpy_arrays')

if not os.path.exists('powerblade_arrays'):
    os.makedirs('powerblade_arrays')

# Get all the csv files names
fnames = glob.glob("PLAID/CSV/*.csv")
files = {} # where we keep the csv files

# Iterate through files parsing and serializing as numpy arrays
print("Parsing files...")

def parse(fname):
    print("  " + str(fname))
    a = np.loadtxt(fname, delimiter=",")

    # save numpy array
    outfilename = "numpy_arrays/" + os.path.basename(fname).split('.')[0]
    np.save(outfilename, a)

    # also downsample the array to match powerblade (42 samples per AC cycle)

    # make x-coordinate array
    xcoors = np.arange(0.0, len(a), 1.0)
    new_xcoors = np.arange(0.0, len(a), (30000/(42*60)))

    # downsample current and voltage
    current = np.rot90(np.split(a, 2, 1)[0])[0]
    downsampled_current = np.interp(new_xcoors, xcoors, current)
    new_current = np.rot90([downsampled_current], axes=(1,0))
    voltage = np.rot90(np.split(a, 2, 1)[1])[0]
    downsampled_voltage = np.interp(new_xcoors, xcoors, voltage)
    new_voltage = np.rot90([downsampled_voltage], axes=(1,0))
    new_array = np.concatenate((new_current, new_voltage), axis=1)

    # save numpy array
    outfilename = "powerblade_arrays/" + os.path.basename(fname).split('.')[0]
    np.save(outfilename, new_array)

p = Pool(50)
p.map(parse,sorted(fnames))
