import os, sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import math

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import get_path_info_v2
from tools.ara_run_manager import file_sorter

Station = int(sys.argv[1])
if Station == 2:
    num_configs = 7
if Station == 3:
    num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/grid_scan_A{Station}_*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

#if Station == 2:
#    num_configs = 7
#if Station == 3:
#    num_configs = 9

m_range = np.linspace(-500, 0, 250 + 1, dtype = float)
d_range = np.linspace(0, 200, 400 + 1, dtype = float)
evt_tot = np.full((3, num_configs), 0, dtype = float)
livesec = np.full((num_configs), 0, dtype = float)
live_days = np.copy(livesec)
dat_cut_hist = np.full((len(d_range), len(m_range), 2, num_configs), 0, dtype = float)
sim_s_cut_hist_indi = np.full((len(d_range), len(m_range), 3, 2, num_configs), 0, dtype = float)
sim_s_pass_hist_indi = np.copy(sim_s_cut_hist_indi)
sim_s_cut_hist = np.full((len(d_range), len(m_range), 2, num_configs), 0, dtype = float)
sim_s_pass_hist = np.copy(sim_s_cut_hist)

pol_type = ['V', 'H']

for r in tqdm(range(len(d_run_tot))):

  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print(d_list[r])
        continue
    #config = int(get_path_info_v2(d_list[r], '_C', '.h5')) - 1
    pol = str(get_path_info_v2(d_list[r], f'A{Station}_', '_R'))   
    pol_idx = pol_type.index(pol)
    print(pol, pol_idx) 

    if pol_idx == 0:
        evt_tot += hf['evt_tot'][:]
        livesec += hf['livesec'][:]
        live_days += hf['live_days'][:]

    dat_cut_hist[:, :, pol_idx] += hf['dat_cut_hist'][:]
    sim_s_cut_hist_indi[:, :, :, pol_idx] += hf['sim_s_cut_hist_indi'][:]
    sim_s_pass_hist_indi[:, :, :, pol_idx] += hf['sim_s_pass_hist_indi'][:]
    sim_s_cut_hist[:, :, pol_idx] += hf['sim_s_cut_hist'][:]
    sim_s_pass_hist[:, :, pol_idx] += hf['sim_s_pass_hist'][:]

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = path+f'grid_scan_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('m_range', data=m_range, compression="gzip", compression_opts=9)
hf.create_dataset('d_range', data=d_range, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('dat_cut_hist', data=dat_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_cut_hist_indi', data=sim_s_cut_hist_indi, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_cut_hist', data=sim_s_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist_indi', data=sim_s_pass_hist_indi, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist', data=sim_s_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.close()

print(file_name)
print('done!')






