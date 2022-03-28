import numpy as np
from scipy.stats import rayleigh
from tqdm import tqdm
from scipy import optimize

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
    ara_root.get_sub_info(Data)


    # wf arr
    wf_len = 1280
    half_wf_len = wf_len // 2
    pad_len = wf_len * 2
    dt = 0.3125
    freq = np.fft.fftfreq(pad_len, dt)
    freq_idx = np.arange(len(freq), dtype = int)
    amp_bins = np.linspace(0, 500, 1000 + 1)
    amp_bin_center = (amp_bins[1:] + amp_bins[:-1]) / 2
    fft_2d = np.full((len(freq), len(amp_bin_center), num_ants), 0, dtype = int)


    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
  
        # get entry and wf
        ara_root.get_entry(evt)
    
        # loop over the antennas
        for ant in range(num_ants):
            real_v = ara_root.get_rf_ch_wf(ant)[1]
            pad_v = np.pad(real_v, (half_wf_len, ), 'constant', constant_values=0)           
            fft_v = np.abs(np.fft.fft(pad_v)) / np.sqrt(len(real_v))
            fft_idx = np.round(fft_v * 2).astype(int)
            fft_2d[freq_idx, fft_idx, ant] += 1

            del real_v, pad_v, fft_v, fft_idx
            ara_root.del_TGraph()
    del ara_root, num_evts
   
    def rayl(x, a):
        return (x/(a*a))*np.exp((-1*(x**2))/(2*a*a))
    mu_idx = np.nanargmax(fft_2d, axis = 1)
    mu_init = amp_bins[:-1][mu_idx]
 
    # rayl. fit
    rayl_mu = np.full((pad_len, num_ants), np.nan, dtype = float)
    for f in tqdm(range(pad_len)):
        for ant in range(num_ants):
            rayl_mu[f, ant] = optimize.curve_fit(rayl, amp_bins[:-1], fft_2d[f, :, ant], p0 = (mu_init[f, ant]), bounds=(0, np.inf))[0]
    del num_ants

    # psd 
    psd = rayl_mu**2 / np.abs(freq[1] - freq[0])

    print('Noise psd collecting is done!')

    return {'freq':freq,
            'rayl_mu':rayl_mu,
            'psd':psd}
    


