PLAID Data
==========

Data from the [PLAID](http://plaidplug.com/) (Plug Load Appliance Identification Dataset).

To download data and convert to numpy arrays:

```
$ ./plaid_serializer.py
```

The folder `numpy_arrays` contains the raw data at 30 kHz sampling rate. The
folder `powerblade_arrays` contains the same data, but downsampled to 2.52 kHz
(PowerBlade's sampling rate) via linear interpolation.

