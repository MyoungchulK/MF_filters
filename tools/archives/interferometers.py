import numpy as np
from scipy.signal import correlate
from scipy.signal import correlation_lags
from scipy.signal import hilbert
from tqdm import tqdm

# custom lib
from tools.antenna import antenna_combination_maker
from tools.wf import interpolation_bin_width

def lag_maker(t_len, dt_ns = interpolation_bin_width()):

    lags = correlation_lags(t_len, t_len, 'full') * dt_ns

    return lags

def cross_correlation(wf, wf_len,
                        corr_len = 'full',
                        corr_type = 'direct',
                        bias_nor = True,
                        unbias_nor = True,
                        hil = True,
                        lag_output = True):

    print('Cross-Correlating WF starts!')

    #pair
    pairs = antenna_combination_maker()[0]    
    pair_len = pairs.shape[0]

    #lag
    pad_t = len(wf[:,0,0])
    lags = lag_maker(pad_t)

    # pre normalization
    if bias_nor == True:
        if unbias_nor == True:
            #01 array
            wf_01 = np.copy(wf)
            wf_01[wf_01 != 0] = 1
        else:
            pass
    
        #rms
        wf_rms = np.nanstd(wf, axis = 0)

        #mean
        wf_mean = np.nanmean(wf, axis = 0)

        #normalization
        wf = np.ma.masked_equal(wf, 0)
        wf -= wf_mean[np.newaxis, :, :]
        wf /= wf_rms[np.newaxis, :, :]
        wf.mask = False
        del wf_mean, wf_rms
    else:
        pass

    # correlation
    evt_num = len(wf[0,0,:])
    corr = np.full((len(lags), pair_len, evt_num), np.nan, dtype=float)
    if unbias_nor == True:
        corr_01 = np.copy(corr)
    else:
        pass

    for evt in tqdm(range(evt_num)):
        for pair in range(pair_len):
            corr[:, pair, evt] = correlate(wf[:, pairs[pair][0], evt], wf[:, pairs[pair][1], evt], corr_len, method=corr_type)
            if unbias_nor == True:
                corr[:, pair, evt] *= wf_len[pairs[pair][0], evt]
                corr_01[:, pair, evt] = correlate(wf_01[:, pairs[pair][0], evt], wf_01[:, pairs[pair][1], evt], corr_len, method=corr_type)
            else:
                pass
    del pairs, pair_len, evt_num

    # post normalization
    if bias_nor == True:
       corr /= pad_t
       if unbias_nor == True:
           del wf_01
           corr /= corr_01
           del corr_01
           # removing nan and inf
           corr[np.isnan(corr)] = 0 #convert x/nan result
           corr[np.isinf(corr)] = 0 #convert nan/nan result 
       else:
           pass
    else:
        pass

    del pad_t

    # hilbert
    if hil == True:
        corr = np.abs(hilbert(corr, axis = 0))
    else:
        pass
    
    print('WF Corss-Correlating is done!')
 
    if lag_output == True:
        return corr, lags
    else:
        del lags
        return corr 

def coval_sampling(corr, lags, table, arr1, arr2,
                    dt_ns = interpolation_bin_width(), 
                    pairs = antenna_combination_maker()[0]):

    #print('CoVal starts!')

    # make the index for pick the corr value
    p0 = ((table - lags[0])/dt_ns).astype(int)

    # check the index wether it is below the 0 for bigger than # of bin
    lag_len = len(lags)
    p0[p0 < 0] = 0
    p0[p0 >= lag_len - 1] = lag_len - 2

    # make the next bin index for performing the interpolation for exact corr value for exact dt
    p1 = p0 + 1

    # array for sampled value
    coval = np.full(table.shape, np.nan, dtype=float)

    # manual interpolation
    # by for loop.....
    for d in range(len(pairs)):
    
        coval[:,:,d] = ((corr[:,d][p1[:,:,d]] - corr[:,d][p0[:,:,d]]) * ((table[:,:,d] - lags[p0[:,:,d]]) / (lags[p1[:,:,d]] - lags[p0[:,:,d]])) + corr[:,d][p0[:,:,d]])
    
    del p0, p1

    # remove csky bin if arrival time was bad (<-100ns)
    coval[(arr1 <-100) | (arr2 <-100)] = 0
   
    #print('CoVal is done!')

    return coval

def sky_map(wf, wf_len, table, arr1, arr2, snr_w = False):

    # correlation
    corr, lags = cross_correlation(wf, wf_len)

    #coval
    coval = coval_sampling(corr, lags, table, arr1, arr2)
    del corr, lags

    # snr weighting

    #sky map
    v_pairs = antenna_combination_maker()[1]
    
    corr_v = np.nansum(coval[:,:,:v_pairs],axis=2)
    corr_h = np.nansum(coval[:,:,v_pairs:],axis=2)
    del coval

    return corr_v, corr_h

















    
