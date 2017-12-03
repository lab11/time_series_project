#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description= 'Find and output last two periods of CV trace')
parser.add_argument('file', metavar='f', type=str, help='numpy data file to process')
parser.add_argument('freq', metavar='x', type=int, help='frequency of data')

args = parser.parse_args()
print(args.file)

data = np.load(args.file)
voltage = data[:,1]
current = data[:,0]
period_len = int(args.freq / 60)
zero_crossings = np.where(np.diff(np.signbit(voltage)))[0]
end = data.shape[0]
for z_cross in np.flip(zero_crossings, 0):
    if voltage[z_cross - 1] < 0:
        end = z_cross
        break;

two_periods = data[z_cross + 1 - period_len * 2: z_cross + 1]
plt.plot(two_periods[:,1])
plt.show()

