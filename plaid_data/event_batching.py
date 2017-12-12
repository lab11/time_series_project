
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

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("numpy_arrays/") and os.path.isdir("PLAID/")):
    print("numpy arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
if not os.path.exists('plaid_paper_batch_data_events'): #different folder for events
    os.makedirs('plaid_paper_batch_data_events')

# collect metadata
metadata_filenames = [("PLAID/meta1.json", 'dataset1'), ("PLAID/meta2StatusesRenamed.json", 'dataset2')]
metadata = {}
for infilename,datasetname in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata[datasetname] = json.load(infile)



throw_out_cnt = 0
for datasetname in sorted(metadata.keys()):
    state_off_cnt = 0   
    for item in metadata[datasetname]:
        # collect various fields
        #print(item)
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
        out_filename = 'plaid_paper_batch_data_events/' + 'event-' + device_class + '-' + device_state + '-' + device_full_name + '-file' + data_id + '.npy' #added event- to the regular filename

        # read input file
        data = np.load(data_filename)
        
        # some defines for readability
        neg = False
        pos = True
        off = False
        on = True
        
        # number of half cycles on either side of the transitions
        num_cycle = 5


        # Gather the data into half cycles
        last_sign = True
        cycles = []
        cur_cycle = []
        first = True
        for row in data:
            cur_cycle.append(list(row))
            #determine the sign
            if row[1] > 0:
                cur_sign = pos
            else:
                cur_sign = neg
            if not cur_sign == last_sign: #if crossed zero
                if cur_sign == pos:
                    if first:
                        first = False
                        cur_cycle = []
                    if not len(cur_cycle) == 0:
                        cycles.append(cur_cycle)
                        cur_cycle = []
                last_sign = cur_sign 
                                
                        

        '''
        #calculate the trigger point
        smart_array = np.array(np.absolute(data[:,0]))
        kmeans = KMeans(n_clusters=2).fit(smart_array.reshape(-1,1))
        high = kmeans.cluster_centers_[0][0]
        low = kmeans.cluster_centers_[1][0]
        #this probably should be done with linear discriminant analysis... 
        #but midpoint is fine........
        trigger = (high+low)/2
        #if (trigger > 1): # this device always draws at least 1 amp... hard to say if it is off... throw it away
        #    print("throwing away due to never having been off: " + data_filename)
        #    continue
        '''
        trigger = 2
        '''
        print(powers)
        plt.plot(data)
        plt.show()
        
        print(data_filename, out_filename)
        exit()
        
        '''
        #extract the data
        off_idxs = []
        on_idxs = []
        last_pwr_state = off
        for idx, cycle in enumerate(cycles):
            if len(cycle) >= 499 and len(cycle) <= 501: 
                tot_power = 0
                for value in cycle:
                    cur_power = abs(value[0]) * abs(value[1])
                    tot_power = tot_power + cur_power
                avg_power = tot_power/len(cycle)
                cur_pwr_state = on
                if avg_power < trigger:
                    cur_pwr_state = off
                if not cur_pwr_state == last_pwr_state: #state change
                    if last_pwr_state == off: #turning on
                        on_idxs.append(idx)
                    else: #turning off
                        off_idxs.append(idx)
                    last_pwr_state = cur_pwr_state
        if len(off_idxs) == 0 and len(on_idxs) == 0: #no state changes found
            print("throwing away due to never having been off: " + data_filename)
            throw_out_cnt = throw_out_cnt + 1
            print(throw_out_cnt)
            continue
        print(data_filename, out_filename, str(len(off_idxs)), str(len(on_idxs)))
    


        #grab data around each state change (this is super gross python)
        final_data = []
        for off_idx in off_idxs: #these index into half cycles
            start_idx = 0
            end_idx = len(cycles)
            if off_idx - num_cycle >= 0:
                start_idx = off_idx - num_cycle
            if off_idx + num_cycle <= len(cycles)-1:
                end_idx = off_idx + num_cycle
            final_data.append(cycles[start_idx:end_idx])
        for on_idx in on_idxs:
            start_idx = 0
            end_idx = len(cycles)
            if on_idx - num_cycle >= 0:
                start_idx = on_idx - num_cycle
            if on_idx + num_cycle <= len(cycles)-1:
                end_idx = on_idx + num_cycle
            final_data.append(cycles[start_idx:end_idx])
        output_data = []
        for cycle_group in final_data:
            for cycle in cycle_group:
                for sample in cycle:
                    output_data.append(sample)
        np.save(out_filename, output_data)
        '''
        
        #select last N full cycles from data
        n_cycles = 1
        frequency = 30000
        voltage = data[:,1]
        current = data[:,0]
        period_len = int(frequency / 60)
        zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]
        end = data.shape[0]
        for z_cross in np.flip(zero_crossings, 0):
            if voltage[z_cross - 1] < 0:
                end = z_cross
                break;
        two_periods = data[z_cross + 1 - period_len * n_cycles: z_cross + 1]
        
        event_col = np.zeros((two_periods.shape[0],1))
        avg_current = np.average(np.absolute(two_periods[:,0]))
        if avg_current < off_trigger_A:
           if state_off_cnt == 0:
               #print("turned_off")
               event_col = np.full((two_periods.shape[0],1),-1)
           state_off_cnt = state_off_cnt + 1
        else:
           if state_off_cnt > 0:
               state_off_cnt = 0
               event_col = np.ones((two_periods.shape[0],1))
               #print("turned_on")
        two_periods = np.hstack((two_periods,event_col))
        output_data = two_periods
        print(output_data.shape)
        '''
        #np.save(out_filename, output_data)
        
