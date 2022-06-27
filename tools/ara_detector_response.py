import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rayleigh
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

def get_rayl_distribution(dat, binning = 1000):

    fft_len = dat.shape[0]
    rfft_2d = np.full((fft_len, binning, num_ants), 0, dtype = int)
    rayl_params = np.full((2, fft_len, num_ants), np.nan, dtype = float)

    if dat.shape[2] == 0:
        rfft_2d = np.full((fft_len, binning, num_ants), np.nan, dtype =float)
        dat_bin_edges = np.full((2, fft_len, num_ants), np.nan, dtype = float)
        return rayl_params, rfft_2d, dat_bin_edges

    dat_bin_edges = np.array([np.nanmin(dat, axis = 2), np.nanmax(dat, axis = 2)], dtype = float)
    dat_bins = np.linspace(dat_bin_edges[0], dat_bin_edges[1], binning + 1, axis = 0)
    dat_half_bin_width = np.abs(dat_bins[1] - dat_bins[0]) / 2

    for freq in tqdm(range(fft_len)):
        for ant in range(num_ants):

            fft_hist = np.histogram(dat[freq, ant], bins = dat_bins[:, freq, ant])[0].astype(int)
            rfft_2d[freq, :, ant] = fft_hist

            mu_init_idx = np.nanargmax(fft_hist)
            if np.isnan(mu_init_idx):
                continue
            mu_init = dat_bins[mu_init_idx, freq, ant] + dat_half_bin_width[freq, ant]
            del fft_hist, mu_init_idx

            try:
                rayl_params[:, freq, ant] = rayleigh.fit(dat[freq, ant], loc = dat_bin_edges[0, freq, ant], scale = mu_init)
            except RuntimeError:
                print(f'Runtime Issue in Freq. {freq} index!')
                pass
            del mu_init
    del dat_bins, dat_half_bin_width, fft_len

    return rayl_params, rfft_2d, dat_bin_edges

class signal_chain_loader:

    def __init__(self, st, freq_range):

        self.st = st
        self.freq_range = freq_range
        self.ohms = 50

    def get_mVperSpGHz_to_dBmperHz(self, dat, use_int = False):

        dBmperHz = 10 * np.log10(dat**2 * 1e-9 / self.ohms / 1e3)
    
        if use_int:
            f = interp1d(ntot_freq, dBmperHz, axis = 0)
            dBmperHz = f(self.freq_range)    

        return dBmperHz

    def get_in_ice_noise_table(self, use_int = False):

        ntot_name = f'../data/in_ice_noise_est/A{self.st}_Ntot_Lab_real.txt'
        print(f'Ntot_path: {ntot_name}')
        ntot_file = np.loadtxt(ntot_name)

        self.ntot_freq = ntot_file[:,0]/1e9
        ntot_dBmperHz = ntot_file[:, 1:]        

        if use_int:
            f = interp1d(self.ntot_freq, ntot_dBmperHz, axis = 0, fill_value = "extrapolate")
            ntot_dBmperHz = f(self.freq_range)
            self.ntot_freq = np.copy(self.freq_range)                 
            del f
        del ntot_name, ntot_file

        return ntot_dBmperHz

    def get_signal_chain(self, dat, use_linear = False):

        psd_dBmperHz = self.get_mVperSpGHz_to_dBmperHz(dat)
        ntot_dBmperHz = self.get_in_ice_noise_table(use_int = True)

        sc_dB = psd_dBmperHz - ntot_dBmperHz

        if use_linear:
            sc_dB = np.sqrt(10**(sc_dB / 10))            
        del psd_dBmperHz, ntot_dBmperHz

        return sc_dB


