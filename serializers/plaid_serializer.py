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
1) Download the PLAID dataset from http://plaidplug.com/ 
2) Unzip the PLAID dataset and put this file in the PLAID folder (were the meta.json files are)
3) python3 plaid_serializer.py

Author: Will Huang
"""

# Make numpy_arrays folder if needed
pathlib.Path('numpy_arrays').mkdir(parents=True, exist_ok=True) 

# Get all the csv files names
os.chdir("CSV")
fnames = glob.glob("*.csv")
files = {} # where we keep the csv files 

""" 
Collect all files in a dict of their filenames
"""
for f in fnames: 
	file = csv.reader(open(f, 'r'), delimiter=",")
	files[f] = file


"""
Serialize numpy arrays 
"""
os.chdir("../numpy_arrays")
for key in files:
	file = files[key]
	dat = []
	for row in file:
		r = [float(x) for x in row] 
		dat.append(r)
	a = np.array(dat)
	np.save(key.split(".")[0], a)

