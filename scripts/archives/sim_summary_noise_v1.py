import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_example_run
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_sim/*noise*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/'

num_evts = 1000
num_ants = 16

sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
coef = np.full((d_len, 2, 2, 2, num_evts), np.nan, dtype = float)
coord = np.full((d_len, 2, 2, 2, 2, num_evts), np.nan, dtype = float) 
coef_max = np.full((d_len, 2, num_evts), np.nan, dtype = float) 
coord_max = np.full((d_len, 2, 3, num_evts), np.nan, dtype = float)
snr = np.full((d_len, num_ants, num_evts), np.nan, dtype = float)
snr_max = np.full((d_len, 2, num_evts), np.nan, dtype = float)
qual = np.full((d_len, num_evts, 4), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)
nan_counts = np.full((d_len), 0, dtype = int)

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2

rad_o = np.array([41, 300], dtype = float)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    coef_tot = hf['coef'][:] # pol, thephi, rad, sol, evt
    coord_tot = hf['coord'][:] # pol, rad, sol, evt
    coef[r] = coef_tot
    coord[r] = coord_tot
    coef_re = np.reshape(coef_tot, (2, 4, -1))
    coef_max_tot = np.nanmax(coef_re, axis = 1)
    coef_max[r] = coef_max_tot
    coord_re = np.reshape(coord_tot, (2, 2, 4, -1))
    counts = 0
    for e in range(num_evts):
        try:
            coef_max_idx = np.nanargmax(coef_re[:, :, e], axis = 1)
        except ValueError:
            counts += 1
            continue
        coord_max[r, 0, :2, e] = coord_re[0, :, coef_max_idx[0], e]
        coord_max[r, 1, :2, e] = coord_re[1, :, coef_max_idx[1], e]
        coord_max[r, 0, 2, e] = rad_o[coef_max_idx[0]//2]
        coord_max[r, 1, 2, e] = rad_o[coef_max_idx[1]//2]
        del coef_max_idx
    nan_counts[r] = counts
    del hf, cons, coef_tot, coord_tot, coef_re, coef_max_tot, coord_re, counts

    hf_name = f'_AraOut.noise_A{Station}_R{config[r]}.txt.run{sim_run[r]}.h5'
    hf = h5py.File(f'{q_path}qual_cut{hf_name}', 'r')
    qual_tot[r] = (hf['tot_qual_cut_sum'][:] != 0).astype(int)
    qual[r] = (hf['tot_qual_cut'][:] != 0).astype(int)
    del hf

    hf = h5py.File(f'{s_path}snr{hf_name}', 'r')
    snr_tot = hf['snr'][:]
    snr[r] = snr_tot
    ex_run = get_example_run(Station, config[r])
    bad_ant = known_issue.get_bad_antenna(ex_run)
    snr_tot[bad_ant] = np.nan
    snr[r, 0] = -np.sort(-snr_tot[:8], axis = 0)[2]
    snr[r, 1] = -np.sort(-snr_tot[8:], axis = 0)[2]
    del hf, snr_tot, ex_run, bad_ant, hf_name

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_Noise_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
hf.create_dataset('snr_max', data=snr_max, compression="gzip", compression_opts=9)
hf.create_dataset('qual', data=qual, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.create_dataset('nan_counts', data=nan_counts, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




