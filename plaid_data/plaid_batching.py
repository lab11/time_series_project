#! /usr/bin/env python3
import numpy as np
import json
import os
import pathlib
import sys
import matplotlib.pyplot as plt

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("numpy_arrays/") and os.path.isdir("PLAID/")):
    print("numpy arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
if not os.path.exists('plaid_paper_batch_data'):
    os.makedirs('plaid_paper_batch_data')

# collect metadata
metadata_filenames = [("PLAID/meta1.json", 'dataset1'), ("PLAID/meta2StatusesRenamed.json", 'dataset2')]
metadata = {}
for infilename,datasetname in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata[datasetname] = json.load(infile)

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

        # determine input and output files
        data_filename = 'numpy_arrays/' + data_id + '.npy'
        out_filename = 'plaid_paper_batch_data/' + device_class + '-' + device_state + '-' + device_full_name + '-file' + data_id + '.npy'

        # read input file
        data = np.load(data_filename)

        #select last N full cycles from data
        n_cycles = 1
        frequency = 30000
        voltage = data[:,1]
        current = data[:,0]
        period_len = int(frequency / 60)
        zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]
        end = data.shape[0]
        periods = []
        counter = 0
        power = np.multiply(voltage,current)
        last_cross = 0
        for z_cross in zero_crossings:
            if not counter % 25:
                if np.mean(power[last_cross:z_cross]) > 5:
                    if(len(data[z_cross + 1 - period_len * n_cycles: z_cross +1]) == 500):
                        periods.append(data[z_cross + 1 - period_len * n_cycles: z_cross +1])
                    else:
                        print("Warning")

            last_cross = z_cross
            counter += 1

        # write output file
        for i in range(0,len(periods)):
            newfname = out_filename[:-4]
            newfname = newfname + "_" + str(i) + ".npy"
            print(data_filename, newfname)
            np.save(newfname, periods[i])

