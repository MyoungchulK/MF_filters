# GainModeling

## Introduction
`gain_calculator.py` is designed to calculate in-situ gain and noise model from event file

1. It will open the `event[xxxxxx].root` file by araroot
2. Selects only Software triggered events
3. Applys interpolation, zero pad, band pass filter, SineSubtract, and rfft to each WF
4. Performs unbinned Rayleigh fitting to rfft array
5. Calculates gain by deconvolving in-ice noise estimation from Rayleigh sigma
6. Saves it in the h5 file

Related documents

1. Gain Modeling Mini-Workshop Intro: https://aradocs.wipac.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2649
2. A23 Noise Modeling: https://aradocs.wipac.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2625
3. SC gain calculation note: https://aradocs.wipac.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2629
4. Gain Workshop: https://aradocs.wipac.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2650

This is example script for modeling gain from ARA data. 

The data that used to make gain model is heavily filtered by band-pass filter and SineSubtract filter. 

Please consider that those two filters are just one of the option user can choose to make cleaner gain model than just using data. It is not absolute requirement.

Please try to develope your own gain script by referring this pacakge:)

## Prerequisites
This script needs 1) hdf5, 2) tqdm, 3) click 4) AraRoot and 5) custom libRootFftwWrapper package

If user is using code in the Madison cluster, 1), 2), and 3) can be installed by `pip`

`pip install hdf5`

`pip install tqdm`

`pip install click`

For AraRoot and libRootFftwWrapper, I recommand user to install these in their own local path

1. AraRoot: https://github.com/ara-software/AraRoot
2. libRootFftwWrapper: https://github.com/MyoungchulK/libRootFftwWrapper Please use this custom package. It is modified to accept different threshold value for each WF. And minimization baoundaries are much wider than original package to reduce wrong minimization.

After install the packages, go to `setup.sh` and change `ARA_UTIL_INSTALL_DIR` to your insyallment directory

`export ARA_UTIL_INSTALL_DIR=/home/mkim/analysis/AraSoft/AraUtil`

to

`export ARA_UTIL_INSTALL_DIR=/path/to/your/dir/`

And type `source setup.sh` in the terminal. It will link your own AraRoot and libRootFftwWrapper

User just can use default `ARA_UTIL_INSTALL_DIR`. But it might not work on future by update

## Example command
Type `python3 gain_calculator.py -d <data path> -p <pedestal path> -o <desired output path> -q <quality cut path. optional>`

Since it is using click package, type `python3 gain_calculator.py --help` also gives you general information

example `python3 gain_calculator.py -d /data/exp/ARA/2014/blinded/L1/ARA02/1026/run004434/event004434.root -p /data/user/mkim/OMF_filter/ARA02/ped_full/ped_full_values_A2_R4434.dat -o /home/mkim/`

quality cut example will be added soon

## Contents of h5 file
1. `freq_range`
2. `num_bins` number of binning for 2d rfft distribution
3. `soft_bin_edges` array of bin edges
4. `soft_rfft_hist2d` rfft distribution in 2d (amplitude vs frequency)
5. `soft_rayl` rayl. fit parameters
6. `soft_sc` array of signal chain gain

