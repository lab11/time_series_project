PLAID Data
==========

Data from the [PLAID](http://plaidplug.com/) (Plug Load Appliance Identification Dataset).

## Getting data

To download data and convert to numpy arrays:

```
$ ./plaid_serializer.py
```

The folder `numpy_arrays` contains the raw data at 30 kHz sampling rate. The
folder `powerblade_arrays` contains the same data, but downsampled to 2.52 kHz
(PowerBlade's sampling rate) via linear interpolation.

## Creating dataset to match NN paper

[Neural Network Ensembles to Real-time Identification of Plug-level Appliance Measurements](https://www.researchgate.net/profile/Karim_Said_Barsim/publication/301771233_Neural_Network_Ensembles_to_Real-time_Identification_of_Plug-level_Appliance_Measurements/links/572718de08ae586b21e1e5f6/Neural-Network-Ensembles-to-Real-time-Identification-of-Plug-level-Appliance-Measurements.pdf)

To create the data batches for training and testing:

```
$ ./plaid_batching.py
```

Each data file is named by its device class, house number, and file number.
This allows a simple `glob` call to collect all files in a class.

