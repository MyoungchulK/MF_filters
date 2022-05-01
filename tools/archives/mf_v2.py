import os, sys
import numpy as np
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from tqdm import tqdm

# custom lib
#from tools.antenna import antenna_info
#from tools.wf import interpolation_bin_width
#from tools.wf import time_pad_maker
#from tools.fft import freq_pad_maker

"""
def lag_pad_maker(dt_ns_scale = False):

    if dt_ns_scale == True:
        dt = interpolation_bin_width()
    else:
        dt = interpolation_bin_width(ns_scale = False)

    off_pad = np.arange(0, time_pad_maker()[1], 1) * dt

    return off_pad, len(off_pad), off_pad[0], off_pad[-1]

def band_square(freq, band_amp, num_temp = 1):
    
    #notch filter
    band_amp[(freq>=430e6) & (freq<=480e6)]=1e-50
    band_amp[(freq<=-430e6) & (freq>=-480e6)]=1e-50
    
    # front band
    band_amp[(freq>=-150e6) & (freq<=150e6),:8]=1e-50
    band_amp[(freq>=-217e6) & (freq<=217e6),8:]=1e-50

    if num_temp == 1:
        band_amp[(freq>=680e6) | (freq<=-680e6)]=1e-50   
 
    if num_temp >= 2:
        band_amp[(freq>=680e6) | (freq<=-680e6),:,0]=1e-50
        band_amp[(freq>=680e6) | (freq<=-680e6),:8,1]=1e-50
        band_amp[(freq>=630e6) | (freq<=-630e6),8:,1]=1e-50

    if num_temp >= 3:
        band_amp[(freq>=665e6) | (freq<=-665e6),:8,2]=1e-50
        band_amp[(freq>=530e6) | (freq<=-530e6),8:,2]=1e-50

    return(band_amp)

def mf_snr(dat, temp, psd, 
            dt = interpolation_bin_width(ns_scale = False), 
            df = freq_pad_maker()[1], 
            hil = False):

    # conjugation
    snr = dat.conjugate()
    snr *= temp[:,:,np.newaxis]
    snr /= psd[:,:,np.newaxis]
    snr = np.abs(2*np.fft.ifft(snr, axis = 0) / dt)

    # normalization
    snr /= np.sqrt(np.abs(2 * np.nansum(temp * temp.conjugate() / psd, axis = 0) * df))[np.newaxis,:,np.newaxis]

    # hilbert application
    if hil == True:
        snr = snr.real
        snr = np.abs(hilbert(snr, axis = 0))
        snr[snr < 0] = 0
    else:
        pass

    print('snr making is done!')

    return snr
        
def mf_max_snr(snr, max_index = False):

    # max picking
    snr_max = np.nanmax(snr, axis = 0)

    if max_index == True:
        # max index
        snr_max_index = np.nanargmax(snr, axis = 0)

        return snr_max, snr_max_index
    else:
        return snr_max

def mf_max_lag(snr_max_index, lag = lag_pad_maker(dt_ns_scale = True)[0], num_Ants = antenna_info()[2]):

    # lag array
    snr_max_lag = np.full(snr_max_index.shape, np.nan)

    # max lag picking
    for ant in range(num_Ants):
        snr_max_lag[ant] = lag[snr_max_index[ant]]    

    return snr_max_lag

def mf_max_snr_lag(dat, temp, psd, hilb = False):

    # anr plot
    snr = mf_snr(dat, temp, psd, hil = hilb)

    # max snr
    snr_max, snr_max_index = mf_max_snr(snr, max_index = True)

    # max snr lag
    snr_max_lag = mf_max_lag(snr_max_index)

    del snr, snr_max_index

    return snr_max, snr_max_lag
"""
    















