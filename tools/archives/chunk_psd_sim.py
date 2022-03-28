import numpy as np
from scipy.stats import rayleigh
from tqdm import tqdm

def psd_sim_collector(Data, Station, Year):

    print('Collecting noise psd starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_constant import ara_const

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    num_evts = ara_root.num_evts

    # wf arr
    pad_len = 1280
    dt = 0.3125
    wf_all = np.full((pad_len * 2, num_ants, 100000), 0, dtype = float)
    wf_len = np.full((num_ants, num_evts), 0, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
  
        # get entry and wf
        ara_root.get_entry(evt)
        
        # loop over the antennas
        for ant in range(num_ants):
            real_v = ara_root.get_rf_ch_wf(ant)[1]
            wf_all[:, ant, evt] = real_v
            wf_len[ant, evt] = len(real_v)
            del real_v
            ara_root.del_TGraph()
    del ara_root, num_evts
    
    # make psd
    freq = np.fft.fftfreq(pad_len, dt)
    fft_all = np.abs(np.fft.fft(wf_all, axis = 0)) / np.sqrt(wf_len)[np.newaxis, :, :]
    del wf_all, wf_len

    print(np.nanmax(fft_all))

    # rayl. fit
    binning = 1000
    fft_max = np.nanmax(fft_all, axis = 2)  
    fft_min = np.nanmin(fft_all, axis = 2) 
    rayl_mu = np.full((pad_len, num_ants), np.nan, dtype = float) 
    for f in tqdm(range(pad_len)):
        for ant in range(num_ants):
            freq_bins = np.linspace(fft_min[f, ant], fft_max[f, ant], 1000 + 1)
            freq_bins_center = (freq_bins[1:] + freq_bins[:-1]) / 2 
            freq_hist = np.histogram(fft_all[f, ant], bins = freq_bins)[0].astype(int)
            mu_idx = np.nanargmax(freq_hist)
            if ~np.isnan(mu_idx):
                mu_freq = freq_bins_center[mu_idx]
                try:
                    loc, scale = rayleigh.fit(fft_all[f, ant], loc = fft_min[f, ant], scale = mu_freq)
                    rayl_mu[f, ant] = loc + scale
                    del loc, scale
                except RuntimeError:
                    #print('Runtime Issue!')
                    pass
                del mu_freq
            del freq_bins, freq_bins_center, mu_idx
    del num_ants, binning, fft_max, fft_min, fft_all

    # psd 
    psd = rayl_mu**2 / np.abs(freq[1] - freq[0])

    print('Noise psd collecting is done!')

    return {'freq':freq,
            'rayl_mu':rayl_mu,
            'psd':psd}
    


