#! /usr/bin/env python3
import numpy as np
import json
import os
import pathlib
import sys
import collections
from sklearn.cluster import KMeans
from sklearn import svm
import matplotlib.pyplot as plt

"""
find state transitions in POWERLBADE DATA
"""

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("powerblade_arrays/") and os.path.isdir("PLAID/")):
    print("powerblade arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
if not os.path.exists('raw_threshold_batch'): #different folder for events
    os.makedirs('raw_threshold_batch')

# collect metadata
metadata_filenames = [("PLAID/meta1.json", 'dataset1'), ("PLAID/meta2StatusesRenamed.json", 'dataset2')]
metadata = {}
for infilename,datasetname in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata[datasetname] = json.load(infile)

no_state_count = 0
for datasetname in sorted(metadata.keys()):
    state_off_cnt = 0   
    for item in metadata[datasetname]:
        # collect various fields
        data_id = str(item['id'])
        location = str(item['meta']['location'])
        device_class = str(item['meta']['type']).replace(' ', '_')
        device_state = str(item['meta']['instances']['status']).replace(' ', '_').replace('-', '_')
        device_appliance = ''
        for app_key in sorted(item['meta']['appliance'].keys()):
            if app_key == 'notes':
                continue
            if device_appliance != '' and item['meta']['appliance'][app_key] != '':
                device_appliance += '_'
            device_appliance += item['meta']['appliance'][app_key].replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace('/', '')
        if device_appliance == '':
            device_appliance = 'unknown'
        device_full_name = datasetname + '_' + location + '_' + device_appliance + '_' + device_class

        # determine input and output files
        data_filename = 'powerblade_arrays/' + data_id + '.npy'
        out_filename = 'raw_threshold_batch/' + 'event-' + device_class + '-' + device_state + '-' + device_full_name + '-file' + data_id + '.npy' #added event- to the regular filename
        outplot = 'raw_threshold_batch_graphs/' + data_id + '.png'
        # read input file
        data = np.load(data_filename)

        # Zero crossing code I stole from neal. Shhhhhhh
        n_cycles = 1
        frequency = 2520
        voltage = data[:,1]
        current = data[:,0]
        period_len = int(frequency / 60)
        zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]

        power_data = [x[0] * x[1] for x in data]
        test = True

        for i in range(0,len(zero_crossings),2):
            if(i+4 >= len(zero_crossings)):
                break
            current_cycle_pwr = np.average(power_data[zero_crossings[i]:zero_crossings[i+2]])
            next_cycle_pwr = np.average(power_data[zero_crossings[i+2]:zero_crossings[i+4]])
            if current_cycle_pwr < 2 and next_cycle_pwr > 2:
                test = False
                break
            if np.average(next_cycle_pwr) >= 2 * np.average(current_cycle_pwr):
                test = False
                break
        if test:
            no_state_count += 1
            print(no_state_count)
            #print("BOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            print("Average power: " + str(np.average(power_data)))
            print("Std Power: " + str(np.std(power_data)))
            outplot = 'raw_threshold_batch_graphs_false/' + data_id + '.png'
        else: 
            

        plt.plot(power_data, 'ro')
        plt.savefig(outplot)
        plt.close()

        #print(data)
        #print(power_data)
        #sys.exit()