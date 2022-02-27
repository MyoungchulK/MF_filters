import numpy as np
from scipy.signal import welch
from scipy.signal import get_window

# custom lib
from tools.wf import time_pad_maker
from tools.wf import interpolation_bin_width

def freq_pad_maker(t_len = time_pad_maker()[1], 
                    dt_s = interpolation_bin_width(ns_scale = False), 
                    cut_tail = False,
                    oneside = False,
                    dfreq = False):

    if oneside == True:
        freq = np.fft.rfftfreq(t_len, dt_s)
        if cut_tail == True:
            freq = freq[:-1]
        else:
            pass
    else:
        freq = np.fft.fftfreq(t_len, dt_s)
    
    if dfreq == True:
        df = np.abs(freq[1]-freq[0])
    else:
        df = None
 
    return freq, df

def fft_maker(wf, wf_dim = 0, 
                oneside = False, 
                symmetry = False, 
                cut_tail = False, 
                absolute = False,
                ortho_norm = None):

    if oneside == True:
        fft = np.fft.rfft(wf, axis = wf_dim)
        if symmetry == True:
            fft *= 2
            if cut_tail == True:
                fft = fft[:-1]
            else:
                pass
        else:
            pass
    else:
        fft = np.fft.fft(wf, axis = wf_dim)

    if ortho_norm is not None:
        fft /= np.sqrt(ortho_norm)[np.newaxis,:,:]
    else:
        pass

    if absolute == True:
        fft = np.abs(fft)
    else:
        pass

    print('FFT conversion is done!')

    return fft

def dbmphz_maker(vsqphz):

    dbmhz = 10 * np.log10(vsqphz / 50) + 30

    return dbmhz

def vsqphz_maker(dbmphz):

    vsqphz = (10 ** ((dbmphz - 30) / 10)) * 50

    return vsqphz

def db_lin_maker(db):

    lin = 10**(db/10)

    return lin

def db_sq_lin_maker(db):

    lin_sq = np.sqrt(db_lin_maker(db))

    return lin_sq

def db_log_maker(lin):

    log = 10 * np.log10(lin)

    return log

def psd_maker(fft, df,
                oneside = False,
                symmetry = False,     
                cut_tail = False, 
                dbm_per_hz = False,
                ortho_norm = None):

    if oneside == True:
        psd = fft * fft.conjugate()
        if symmetry == True:
            psd *= 2
            if cut_tail == True:
                psd = psd[:-1]
            else:
                pass
        else:
            pass
    else:
        fft_flip = fft[1:-1]
        fft_full = np.append(fft,fft_flip[::-1],axis=0)
        del fft_flip
        psd = fft_full * fft_full.conjugate()
        del fft_full

    # V^2/Hz
    psd /= df

    if ortho_norm is not None:
        psd /= ortho_norm[np.newaxis,:]
    else:
        pass

    if dbm_per_hz == True:
        psd = dbmphz_maker(psd)
    else:
        pass

    return psd

def welch_maker(wf, wf_dim = 0, 
                ndf = 1/interpolation_bin_width(ns_scale = False), 
                t_len = time_pad_maker()[1], 
                win_func_welch = 'rectangular',
                onside = True,
                dbm_per_hz = False,
                pad_norm = None):

    psd = welch(wf, axis=wf_dim, fs=ndf, nperseg=t_len,
                    detrend=None, return_onesided=onside,
                    window=get_window(win_func_welch, t_len, fftbins=False))[1]

    if pad_norm is not None:
        psd *= t_len
        psd /= pad_norm[np.newaxis,:]
    else:
        pass

    if dbm_per_hz == True:
        psd = dbmphz_maker(psd)
    else:
        pass

    return psd
 



