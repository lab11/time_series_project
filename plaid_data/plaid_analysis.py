#! /usr/bin/env python3
import os
import sys
import json

# check if plaid dataset exists
if not (os.path.exists("PLAID/") and os.path.isdir("PLAID/")):
    print("PLAID not downloaded yet. Run `plaid_serializer.py`")
    sys.exit()

metadata_filenames = ["PLAID/meta1.json",
                      #"PLAID/meta2.json",
                      "PLAID/meta2StatusesRenamed.json"]

# iterate through metadata files and each JSON blob in them
for infilename in sorted(metadata_filenames):
    print('\n\n' + infilename)
    locations = []
    status_types = []
    device_types = {}
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

        print(" - " + item + space + "(count " + str(device_types[item]['count']) + ",\t number of locs " + str(device_types[item]['unique']) + ")")
        print("\t" + str(device_types[item]['statuses']))


    # special testing to answer some validity questions
    if False:
        dev_dict = {}
        for loc in locations:
            dev_dict[loc] = {}
            for item in metadata:
                if item['meta']['location'] != loc:
                    continue

                dev_type = item['meta']['type']
                if dev_type not in dev_dict[loc].keys():
                    dev_dict[loc][dev_type] = {}

                dev_appliance = ''
                for app_key in sorted(item['meta']['appliance'].keys()):
                    if app_key == 'notes':
                        continue
                    if dev_appliance != '' and item['meta']['appliance'][app_key] != '':
                        dev_appliance += '_'
                    dev_appliance += item['meta']['appliance'][app_key].replace(' ', '_').replace('-', '_').replace('.', '_').replace('(', '').replace(')', '').replace('/', '')

                if dev_appliance not in dev_dict[loc][dev_type]:
                    dev_dict[loc][dev_type][dev_appliance] = []

                dev_dict[loc][dev_type][dev_appliance].append(int(item['id']))

        for loc in sorted(dev_dict.keys()):
            print(loc)
            for dev_type in sorted(dev_dict[loc].keys()):
                print(' ' + dev_type)
                for dev_appliance in sorted(dev_dict[loc][dev_type].keys()):
                    ids = ''
                    prev_id = 0
                    for dev_id in sorted(dev_dict[loc][dev_type][dev_appliance]):
                        if prev_id > 0 and dev_id != prev_id+1:
                            # Note, this was tested and never actually occurs
                            ids += '<-> '
                        ids += str(dev_id) + ' '
                    special = ''
                    if len(dev_dict[loc][dev_type][dev_appliance]) > 6:
                        special = ' RATHER LONG!!'
                    print('  ' + str(dev_appliance) + ' ' + str(ids) + special)

