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
power-event - instantaneous power 
"""

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("numpy_arrays/") and os.path.isdir("PLAID/")):
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
transition_classes = {}
no_transition_classes = {}
no_counter = 0
no_classes = {}
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
        data_filename = 'numpy_arrays/' + data_id + '.npy'
        out_filename = 'raw_threshold_batch/' + 'numpy-power-event-' + device_class + '-' + device_state + '-' + device_full_name + '-file' + data_id + '.npy' #added event- to the regular filename
        # read input file
        data = np.load(data_filename)

        # Zero crossing code I stole from neal. Shhhhhhh
        n_cycles = 1
        frequency = 30000
        voltage = data[:,1]
        current = data[:,0]
        period_len = int(frequency / 60)
        zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]

        power_data = [x[0] * x[1] for x in data]
        no_transition = True
        transition_point = 0

        for i in range(0,len(zero_crossings),2):
            if(i+4 >= len(zero_crossings)):
                break
            current_cycle_pwr = np.average(power_data[zero_crossings[i]:zero_crossings[i+2]])
            next_cycle_pwr = np.average(power_data[zero_crossings[i+2]:zero_crossings[i+4]])
            if current_cycle_pwr < 2 and next_cycle_pwr > 2:
                transition_point = zero_crossings[i+2]
                no_transition = False
                break
            if np.average(next_cycle_pwr) >= 2 * np.average(current_cycle_pwr):
                transition_point = zero_crossings[i+2]
                no_transition = False
                break
        if no_transition:
            no_state_count += 1
            print(no_state_count)
            print("Average power: " + str(np.average(power_data)))
            print("Std Power: " + str(np.std(power_data)))
            if device_class not in no_transition_classes:
                no_transition_classes[device_class] = 0
            no_transition_classes[device_class] += 1

        else: 
            if device_class not in transition_classes:
                transition_classes[device_class] = 0
            transition_classes[device_class] += 1
            end_point = transition_point + 15000 # Grab 0.5 seconds of data
            if end_point >= len(power_data):
                print("NOOOOOO")
                print(out_filename)
                print(end_point)
                print(len(power_data))
                no_counter += 1
                if device_class not in no_classes:
                    no_classes[device_class] = 0
                no_classes[device_class] += 1
            else:
                output_data = power_data[transition_point:end_point]
                np.save(out_filename, output_data)

        #print(data)
        #print(power_data)
        #sys.exit()

print("Transition Classes")
print(transition_classes)
print("-----------")
print(no_transition_classes)
print("--------------")
print(no_classes)
print(no_counter)