#! /usr/bin/env python3
import numpy as np
import json
import os
import pathlib
import sys

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("powerblade_arrays/") and os.path.isdir("PLAID/")):
    print("powerblade arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
out_dir = 'powerblade_data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# collect metadata
metadata_filenames = [("PLAID/meta1.json", 'dataset1'), ("PLAID/meta2StatusesRenamed.json", 'dataset2')]
metadata = {}
input_traces = 0
for infilename,datasetname in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata[datasetname] = json.load(infile)
        input_traces += len(metadata[datasetname])

# determine output size based on data characteristics
frequency = 42*60
num_traces = input_traces
data_len = int((frequency/60 * 2) + 2) # voltage & current, plus name and class
output_data = np.zeros((num_traces, data_len), dtype=float)

# lookup arrays for device name, class, and house names
name_map = []
class_map = []

# create maps
for datasetname in sorted(metadata.keys()):
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

        # map device names
        if device_full_name not in name_map:
            name_map.append(device_full_name)

        # map device classes
        if device_class not in class_map:
            class_map.append(device_class)

name_map.sort()
class_map.sort()

# iterate through data
output_index = 0
for datasetname in sorted(metadata.keys()):
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

        # read input file
        data_filename = 'powerblade_arrays/' + data_id + '.npy'
        data = np.load(data_filename)
        print('Processing ' + data_filename)

        #select last N full cycles from data
        n_cycles = 2
        voltage = data[:,1]
        current = data[:,0]
        period_len = int(frequency / 60)
        zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]
        end = data.shape[0]
        for z_cross in np.flip(zero_crossings, 0):
            if voltage[z_cross - 1] < 0:
                end = z_cross
                break
        two_periods = data[z_cross + 1 - period_len * n_cycles: z_cross + 1]

        output_data[output_index, :-2] = two_periods[0:period_len].flatten(order='F')
        output_data[output_index, -2]  = name_map.index(device_full_name)
        output_data[output_index, -1]  = class_map.index(device_class)
        output_index += 1

# save data
np.save(out_dir + 'powerblade_traces_bundle', output_data)
np.save(out_dir + 'powerblade_traces_class_map', class_map)
np.save(out_dir + 'powerblade_traces_name_map', name_map)

