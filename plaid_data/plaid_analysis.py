#! /usr/bin/env python3
import os
import sys
import json
import pprint

# check if plaid dataset exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()

metadata_filenames = ["PLAID/meta1.json", "PLAID/meta2.json"]

locations = []
status_types = []
device_types = {}
# iterate through metadata files and each JSON blob in them
for infilename in metadata_filenames:
    with open(infilename, 'r') as infile:
        metadata = json.load(infile)
        for item in metadata:
            # store data in a bunch of dicts!

            if item['meta']['location'] not in locations:
                locations.append(item['meta']['location'])

            if item['meta']['instances']['status'] not in status_types:
                status_types.append(item['meta']['instances']['status'])

            if item['meta']['type'] not in device_types.keys():
                device_types[item['meta']['type']] = {}
                device_types[item['meta']['type']]['count'] = 0
                device_types[item['meta']['type']]['locations'] = []
                device_types[item['meta']['type']]['statuses'] = []
            device_types[item['meta']['type']]['count'] += 1

            if item['meta']['location'] not in device_types[item['meta']['type']]['locations']:
                device_types[item['meta']['type']]['locations'].append(item['meta']['location'])

            if item['meta']['instances']['status'] not in device_types[item['meta']['type']]['statuses']:
                device_types[item['meta']['type']]['statuses'].append(item['meta']['instances']['status'])



print("")
print("Locations: " + str(len(locations)))

print("")
print("Status Types: " + str(len(status_types)))

print("")
print("Unique device types:  (count " + str(len(device_types)) + ")")
for item in device_types.keys():

    # calculate unique locations for each device
    device_types[item]['unique'] = len(device_types[item]['locations'])


    # spacing to make the text line up
    space = "\t\t\t\t"
    if len(item) > 4:
        space = "\t\t\t"
    if len(item) > 12:
        space = "\t\t"
    if len(item) > 16:
        space = "\t"

    print(" - " + item + space + "(count " + str(device_types[item]['count']) + ",\t unique " + str(device_types[item]['unique']) + ")")
    print("\t" + str(device_types[item]['statuses']))

