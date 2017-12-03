#! /usr/bin/env python3
import numpy as np
import json
import os
import pathlib

# check that the PLAID dataset already exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()
if not (os.path.exists("numpy_arrays/") and os.path.isdir("PLAID/")):
    print("numpy arrays not created yet. Run `plaid_serializer.py`")
    sys.exit()

# make folders if needed
pathlib.Path('plaid_paper_batch_data').mkdir(parents=True, exist_ok=True)

# collect metadata
metadata_filenames = ["PLAID/meta1.json", "PLAID/meta2.json"]
metadata = []
for infilename in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata += json.load(infile)

for item in metadata:
    # determine input and output files
    data_id = str(item['id'])
    location = str(item['meta']['location'])
    device_class = str(item['meta']['type']).replace(' ', '_')
    data_filename = 'numpy_arrays/' + data_id + '.npy'
    out_filename = 'plaid_paper_batch_data/' + device_class + '-' + location + '-file' + data_id + '.npy'
    print(data_filename, out_filename)

    # read input file
    data = np.load(data_filename)
    
    #XXX: select last two full cycles from data
    output_data = np.array([]) # delete me

    # write output file
    np.save(out_filename, output_data)

