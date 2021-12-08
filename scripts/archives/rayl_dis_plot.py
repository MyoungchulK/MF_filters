import numpy as np
import os, sys
import h5py
from tqdm import tqdm

#custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.plot import hist_2d
from tools.plot import imshow_2d
from tools.antenna import antenna_info

def hist_prep(freq, freq_dim_len, dat):

    freq_2d = np.repeat(freq[:,np.newaxis], freq_dim_len, axis=1)
    dat_2d = np.copy(dat)
    freq_2d = freq_2d[~np.isnan(dat_2d)]
    dat_2d = dat_2d[~np.isnan(dat_2d)]

    return freq_2d, dat_2d

Station = int(sys.argv[1])
d_path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Rayl_tot/'
os.chdir(d_path)
h5_file = f'Rayleigh_Fit_A{Station}_tot.h5'
print(f'opening {h5_file}...')
hf = h5py.File(h5_file, 'r')

print('assinging data...')
run_index = hf['run_range'][:]
freq = hf['freq'][:]/1e9
good_runs_index = hf['good_runs_index'][:]
print(good_runs_index)
good_runs_index_test = np.copy(good_runs_index)
print(len(good_runs_index_test))
print(len(good_runs_index_test[~np.isnan(good_runs_index_test)]))
del good_runs_index_test
loc_w_sc = hf['loc_w_sc'][:] * good_runs_index[np.newaxis, np.newaxis, :]
scale_w_sc = hf['scale_w_sc'][:] * good_runs_index[np.newaxis, np.newaxis, :]
loc_wo_sc = hf['loc_wo_sc'][:] * good_runs_index[np.newaxis, np.newaxis, :]
scale_wo_sc = hf['scale_wo_sc'][:] * good_runs_index[np.newaxis, np.newaxis, :]
mu_w_sc = loc_w_sc + scale_w_sc 
mu_wo_sc = loc_wo_sc + scale_wo_sc 

print(freq.shape)
print(good_runs_index.shape)
print(loc_w_sc.shape)
print(scale_w_sc.shape)
print(loc_wo_sc.shape)
print(scale_wo_sc.shape)
print(mu_w_sc.shape)
print(mu_wo_sc.shape)
del hf

print('log conversion...')
loc_w_sc = np.log10(loc_w_sc)
scale_w_sc = np.log10(scale_w_sc)
loc_wo_sc = np.log10(loc_wo_sc)
scale_wo_sc = np.log10(scale_wo_sc)
mu_w_sc = np.log10(mu_w_sc)
mu_wo_sc = np.log10(mu_wo_sc)
print('loc_w_sc',np.nanmin(loc_w_sc),np.nanmax(loc_w_sc))
print('scale_w_sc',np.nanmin(scale_w_sc),np.nanmax(scale_w_sc))
print('loc_wo_sc',np.nanmin(loc_wo_sc),np.nanmax(loc_wo_sc))
print('scale_wo_sc',np.nanmin(scale_wo_sc),np.nanmax(scale_wo_sc))
print('mu_w_sc',np.nanmin(mu_w_sc),np.nanmax(mu_w_sc))
print('mu_wo_sc',np.nanmin(mu_wo_sc),np.nanmax(mu_wo_sc))

Antenna = antenna_info()[0]
freq_dim_len = len(loc_w_sc[0,0,:])
#run_index = np.arange(len(good_runs_index))
df = np.abs(freq[1] - freq[0])
del good_runs_index

for a in tqdm(range(len(Antenna))):
    
    imshow_2d(r'Rayleigh Fit, Loc w/ SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Loc_w_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = loc_w_sc[::-1,a]
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)
    
    freq_2d, loc_w_sc_2d = hist_prep(freq, freq_dim_len, loc_w_sc[:,a])
    hist_2d(r'Rayleigh Fit, Loc w/ SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Loc_w_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-22,-6,0.01)
            , x_dat = freq_2d, y_dat = loc_w_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -14, ymax = -9)
    del loc_w_sc_2d

    imshow_2d(r'Rayleigh Fit, Loc w/o SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Loc_wo_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = loc_wo_sc[::-1,a]
            , cmin = -16, cmax = -13
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)

    freq_2d, loc_wo_sc_2d = hist_prep(freq, freq_dim_len, loc_wo_sc[:,a])
    hist_2d(r'Rayleigh Fit, Loc w/o SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Loc_wo_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-22,-6,0.01)
            , x_dat = freq_2d, y_dat = loc_wo_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -17, ymax = -12)
    del loc_wo_sc_2d

    imshow_2d(r'Rayleigh Fit, Scale w/ SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Scale_w_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = scale_w_sc[::-1,a]
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)

    freq_2d, scale_w_sc_2d = hist_prep(freq, freq_dim_len, scale_w_sc[:,a])
    hist_2d(r'Rayleigh Fit, Scale w/ SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Scale_w_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-22,-6,0.01)
            , x_dat = freq_2d, y_dat = scale_w_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -11, ymax = -9)
    del scale_w_sc_2d

    imshow_2d(r'Rayleigh Fit, Scale w/o SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Scale_wo_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = scale_wo_sc[::-1,a]
            , cmin = -13.2, cmax = -12.6
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)

    freq_2d, scale_wo_sc_2d = hist_prep(freq, freq_dim_len, scale_wo_sc[:,a])
    hist_2d(r'Rayleigh Fit, Scale w/o SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Scale_wo_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-22,-6,0.01)
            , x_dat = freq_2d, y_dat = scale_wo_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -13.5, ymax = -12.5)
    del scale_wo_sc_2d

    imshow_2d(r'Rayleigh Fit, Mu w/ SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Mu_w_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = mu_w_sc[::-1,a]
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)

    freq_2d, mu_w_sc_2d = hist_prep(freq, freq_dim_len, mu_w_sc[:,a])
    hist_2d(r'Rayleigh Fit, Mu w/ SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Mu_w_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-22,-6,0.01)
            , x_dat = freq_2d, y_dat = mu_w_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -11, ymax = -9)
    del mu_w_sc_2d
    
    imshow_2d(r'Rayleigh Fit, Mu w/o SC, ' + Antenna[a]
            , r'Run index', r'Frequency [ $GHz$ ]'
            , d_path
            , 'Rayleigh_Fit_Mu_wo_SC_' + Antenna[a] + '_runs.png'
            , cmap_c = 'jet'
            , cbar_legend = r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , x_range = run_index, y_range = freq
            , x_width = 0.5, y_width = df
            , xy_dat = mu_wo_sc[::-1,a]
            , cmin = -13, cmax = -12.6
            , xmin = run_index[0]-0.5, xmax = run_index[-1]+0.5
            , ymin = freq[0] - df, ymax = freq[-1] + df)
    
    freq_2d, mu_wo_sc_2d = hist_prep(freq, freq_dim_len, mu_wo_sc[:,a])
    hist_2d(r'Rayleigh Fit, Mu w/o SC, ' + Antenna[a]
            , r'Frequency [ $GHz$ ]',r'Amplitude [ $log_{10}(V/Hz)$ ]'
            , d_path
            , 'Rayleigh_Fit_Mu_wo_SC_' + Antenna[a] + '.png'
            , cmap_c = 'viridis'
            , cbar_legend = r'# of Runs'
            , x_range = freq, y_range = np.arange(-13,-9,0.002)
            , x_dat = freq_2d, y_dat = mu_wo_sc_2d
            , xmin = 0, xmax = 1
            , ymin = -13, ymax = -12.5)
    del mu_wo_sc_2d, freq_2d
    
del Antenna, freq_dim_len, freq, loc_w_sc, loc_wo_sc, scale_w_sc, scale_wo_sc, mu_w_sc, mu_wo_sc
print('done!')

