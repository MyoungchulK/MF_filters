import numpy as np
from scipy.signal import welch
from scipy.signal import get_window

def freq_pad_maker(t_len, dt):

    freq = np.fft.fftfreq(t_len, dt)

    return freq, np.abs(freq[1]-freq[0])

def welch_maker(wf, ndf, wf_len, win_func_welch = 'rectangular'):

    psd_f, psd_v_sq = welch(wf, axis=0, fs=ndf, nperseg=wf_len
                            , detrend=None, return_onesided=False
                            , window=get_window(win_func_welch, wf_len, fftbins=False))

    return psd_f, psd_v_sq*2

def pad_psd_maker(wf, ndf, wf_len_p, win_func_pad = 'rectangular'):

    return welch_maker(wf, ndf, wf_len_p, win_func_welch = win_func_pad)[1]

def psd_maker(wf, ndf, wf_len_p, wf_len, win_func = 'rectangular'):

    return pad_psd_maker(wf, ndf, wf_len_p, win_func_pad = win_func) * wf_len_p / wf_len[np.newaxis, :]




