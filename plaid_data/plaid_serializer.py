#! /usr/bin/env python3
import numpy as np
import csv 
import glob
import os
import pathlib


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

# Make numpy_arrays folder if needed
pathlib.Path('numpy_arrays').mkdir(parents=True, exist_ok=True) 

# Get all the csv files names
fnames = glob.glob("PLAID/CSV/*.csv")
files = {} # where we keep the csv files 

# Iterate through files parsing and serializing as numpy arrays
for f in fnames:
    with open(f, 'r') as infile:
        datfile = csv.reader(infile, delimiter=",")
        dat = []
        for row in datfile:
            r = [float(x) for x in row] 
            dat.append(r)
        a = np.array(dat)
        outfilename = "numpy_arrays/" + os.path.basename(f).split('.')[0]
        np.save(outfilename, a)

