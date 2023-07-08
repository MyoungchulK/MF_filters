import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Type = str(sys.argv[2])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

q_name = ['noise trigger', 'cw ratio', 'calpuler cut', 'surface corr cut', 'surface mf cut', 'op antenna cut']
q_len = len(q_name)
print(q_len)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/*{Type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
config = np.full((d_len, 4), 0, dtype = int) # sim run, config, flavor, energy
evt_rate = np.full((d_len, num_evts), np.nan, dtype = float)
qual_indi = np.full((d_len, num_evts, q_len), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)

energys = np.array([16, 17, 18, 19, 20], dtype = int)
en_bin_center = np.array([16.5, 17.5, 18.5, 19.5, 20.5], dtype = float)

sig_eff_tot = np.full((num_configs, 3, 3), 0, dtype = float) # config, flavor, eff type
sig_eff_energy_tot = np.full((num_configs, len(energys), 3, 3), 0, dtype = float) # config, energy, flavor, eff type
sig_eff_indi = np.full((num_configs, q_len, 3, 3), 0, dtype = float) # config, qual type, flavor, eff type
sig_eff_energy_indi = np.full((num_configs, len(energys), q_len, 3, 3), 0, dtype = float) # config, energy, qual type, flavor, eff type

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    con = hf['config'][:]
    cons = np.array([con[1], con[2] - 1, con[4] - 1, con[5]], dtype = int)
    config[r, 0] = cons[0] # sim run
    config[r, 1] = cons[1] # config
    config[r, 2] = cons[2] # flavor
    config[r, 3] = cons[3] # energy

    evt_rates = hf['evt_rate'][:]
    evt_rate[r] = evt_rates
    tot_qual_cuts = hf['tot_qual_cut'][:] != 0
    tot_qual_cut_sums = hf['tot_qual_cut_sum'][:] != 0
    qual_indi[r] = tot_qual_cuts
    qual_tot[r] = tot_qual_cut_sums

    print(evt_rates.shape, tot_qual_cut_sums.shape)
    tot_rate = np.nansum(evt_rates)
    tot_rate_good = np.nansum(evt_rates[~tot_qual_cut_sums])
    tot_rate_bad = np.nansum(evt_rates[tot_qual_cut_sums])
    tot_r = np.array([tot_rate, tot_rate_good, tot_rate_bad])

    evt_ep = np.repeat(evt_rates[:, np.newaxis], q_len, axis = 1)
    evt_good = np.copy(evt_ep)
    evt_good[tot_qual_cuts] = np.nan
    evt_bad = np.copy(evt_ep)
    evt_bad[~tot_qual_cuts] = np.nan
    evt_ep = np.nansum(evt_ep, axis = 0)    
    evt_good = np.nansum(evt_good, axis = 0)
    evt_bad = np.nansum(evt_bad, axis = 0)
    del evt_rates, tot_qual_cuts, tot_qual_cut_sums

    sig_eff_tot[cons[1], cons[2]] += tot_r
    sig_eff_energy_tot[cons[1], int(cons[3] - 16), cons[2]] += tot_r
    sig_eff_indi[cons[1], :, cons[2], 0] += evt_ep
    sig_eff_indi[cons[1], :, cons[2], 1] += evt_good
    sig_eff_indi[cons[1], :, cons[2], 2] += evt_bad
    sig_eff_energy_indi[cons[1], int(cons[3] - 16), :, cons[2], 0] += evt_ep
    sig_eff_energy_indi[cons[1], int(cons[3] - 16), :, cons[2], 1] += evt_good
    sig_eff_energy_indi[cons[1], int(cons[3] - 16), :, cons[2], 2] += evt_bad
    del con, cons, tot_rate, tot_rate_good, tot_rate_bad, tot_r, evt_ep, evt_good, evt_bad

per_tot = sig_eff_tot / sig_eff_tot[:, :, 0][:, :, np.newaxis] * 100
for c in range(num_configs):
    print(f'tot config {int(c + 1)}: {np.round(per_tot[c, :, 2], 2)}')
print()

per_indi = np.nanmean(sig_eff_indi / sig_eff_indi[:, :, :, 0][:, :, np.newaxis] * 100, axis = 2)
for c in range(num_configs):
    print(f'indi config {int(c + 1)}: {np.round(per_indi[c, :, 2], 2)}')
print()

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sig_Eff_Check_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('qual_indi', data=qual_indi, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.create_dataset('en_bin_center', data=en_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sig_eff_tot', data=sig_eff_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_eff_energy_tot', data=sig_eff_energy_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_eff_indi', data=sig_eff_indi, compression="gzip", compression_opts=9)
hf.create_dataset('sig_eff_energy_indi', data=sig_eff_energy_indi, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






