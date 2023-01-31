import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rayleigh
from tqdm import tqdm

# custom lib
from tools.ara_constant import ara_const

ara_const = ara_const()
num_ants = ara_const.USEFUL_CHAN_PER_STATION

def get_rayl_distribution(dat, binning = 100):

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
                dat_amp = dat[freq, ant][~np.isnan(dat[freq, ant])]
                rayl_params[:, freq, ant] = rayleigh.fit(dat_amp, loc = dat_bin_edges[0, freq, ant], scale = mu_init)
                del dat_amp
            except RuntimeError:
                print(f'Runtime Issue in Freq. {freq} index!')
                rayl_params[0, freq, ant] = 0
                rayl_params[1, freq, ant] = mu_init
                pass
            del mu_init
    del dat_bins, dat_half_bin_width, fft_len

    return rayl_params, rfft_2d, dat_bin_edges

def get_signal_chain_gain(soft_rayl, freq_range, dt, st):

    p1 = soft_rayl / 1e3 * np.sqrt(1e-9) # mV to V and ns to s
    print(p1.shape)
    freq_mhz = freq_range * 1e3 # GHz to MHz

    h_tot_path = f'../data/sc_info/A{st}_Htot.txt'
    print('h_tot_path:', h_tot_path)
    h_tot = np.loadtxt(h_tot_path)
    f = interp1d(h_tot[:,0], h_tot[:, 1:], axis = 0, fill_value = 'extrapolate')
    h_tot_int = f(freq_mhz)
    del freq_mhz, h_tot, f 

    Htot = h_tot_int * np.sqrt(dt * 1e-9)
    Hmeas = p1 * np.sqrt(2) # power symmetry
    Hmeas *= np.sqrt(2) # surf_turf
    soft_sc = Hmeas / Htot
    del p1, Htot, Hmeas

    return soft_sc

def get_rayl_bad_run(soft_len, rayl_nan, st, run, analyze_blind_dat = False, verbose = False):

    if analyze_blind_dat:
        if soft_len == 0 or rayl_nan:
            if verbose:
                print(f'A{st} R{run} is bad for noise modeling!!!')
            bad_dir = f'../data/rayl_runs/'
            if not os.path.exists(bad_dir):
                os.makedirs(bad_dir)
            bad_name = f'rayl_run_A{st}.txt'
            bad_path = f'{bad_dir}{bad_name}'
            bad_run_info = f'{run}\n'
            if os.path.exists(bad_path):
                if verbose:
                    print(f'There is {bad_path}')
                bad_run_arr = []
                with open(bad_path, 'r') as f:
                    for lines in f:
                        run_num = int(lines)
                        bad_run_arr.append(run_num)
                bad_run_arr = np.asarray(bad_run_arr, dtype = int)
                if run in bad_run_arr:
                    if verbose:
                        print(f'Run{run} is already in {bad_path}!')
                    else:
                        pass
                else:
                    if verbose:
                        print(f'Add run{run} in {bad_path}!')
                    with open(bad_path, 'a') as f:
                        f.write(bad_run_info)
                del bad_run_arr
            else:
                if verbose:
                    print(f'There is NO {bad_path}')
                    print(f'Add run{run} in {bad_path}!')
                with open(bad_path, 'w') as f:
                    f.write(bad_run_info)
            del bad_path, bad_run_info, bad_dir, bad_name
            bad_run = np.array([1], dtype = int)
        else:
            bad_run = np.array([0], dtype = int)
    else:
        bad_run = np.full((1), np.nan, dtype = float)

    return bad_run
















