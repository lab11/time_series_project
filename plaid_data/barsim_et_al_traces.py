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
if not (os.path.exists("numpy_arrays/") and os.path.isdir("PLAID/")):
    print("numpy arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
out_dir = 'barsim_et_al_data/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# collect metadata
metadata_filenames = [("PLAID/meta1.json", 'dataset1')] # only do dataset1 for now
metadata = {}
input_traces = 0
for infilename,datasetname in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata[datasetname] = json.load(infile)
        input_traces += len(metadata[datasetname])

# determine output size based on data characteristics
phased_trace_count = 50
frequency = 30000
num_traces = input_traces * phased_trace_count # number of traces * duplicates (with phase offset)
data_len = int((frequency/60 * 2) + 3) # voltage & current, plus name, class and location
output_data = np.zeros((num_traces, data_len), dtype=float)

# lookup arrays for device name, class, and house names
name_map = []
class_map = []
house_map = []

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

        # map device locations
        if location not in house_map:
            house_map.append(location)
name_map.sort()
class_map.sort()
house_map.sort()

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
        data_filename = 'numpy_arrays/' + data_id + '.npy'
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

        # create N phase-shifted "traces" out of that
        traces = np.zeros((phased_trace_count, data_len-3))
        phase_step = int((frequency/60)/phased_trace_count)
        for step,trace_row in enumerate(traces):
            offset = step*phase_step
            trace_row[:] = two_periods[offset: period_len+offset, :].flatten(order='F')

        # write traces to output data
        for trace_row in traces:
            output_data[output_index, :-3] = trace_row
            output_data[output_index, -3] = house_map.index(location)
            output_data[output_index, -2] = name_map.index(device_full_name)
            output_data[output_index, -1] = class_map.index(device_class)
            output_index += 1

# save data
np.save(out_dir + 'traces_bundle', output_data)
np.save(out_dir + 'traces_class_map', class_map)
np.save(out_dir + 'traces_name_map', name_map)
np.save(out_dir + 'traces_house_map', house_map)

