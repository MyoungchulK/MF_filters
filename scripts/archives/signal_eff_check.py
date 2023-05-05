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

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*signal*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/'

num_evts = 100
pnu = np.full((d_len, num_evts), np.nan, dtype = float)
sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
flavor = np.copy(sim_run)
exponent = np.full((d_len, 2), 0, dtype = int)
snr = np.copy(pnu)
qual = np.full((d_len, num_evts, 4), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)
evt_rate = np.copy(pnu)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  #if r >= count_i and r < count_ff:

    hf = h5py.File(d_list[r], 'r')
    pnu[r] = hf['pnu'][:]
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    flavor[r] = hf['nuflavorint'][0]
    exponent[r] = hf['exponent_range'][:]
    del hf, cons
    
    hf_name = f'_AraOut.signal_E{int(exponent[r, 0])}_F{flavor[r]}_A{Station}_R{config[r]}.txt.run{sim_run[r]}.h5'
    hf = h5py.File(f'{s_path}snr{hf_name}', 'r')
    snr_tot = hf['snr'][:]
    ex_run = get_example_run(Station, config[r])
    bad_ant = known_issue.get_bad_antenna(ex_run)
    snr_tot[bad_ant] = np.nan
    snr_tot = -np.sort(-snr_tot, axis = 0)
    snr[r] = snr_tot[2]
    del hf, snr_tot, ex_run, bad_ant

    hf = h5py.File(f'{q_path}qual_cut{hf_name}', 'r')
    qual_tot[r] = (hf['tot_qual_cut_sum'][:] != 0).astype(int)
    qual[r] = (hf['tot_qual_cut'][:] != 0).astype(int)
    evt_rate[r] = hf['evt_rate'][:]
    del hf_name, hf

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Signal_Eff_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
hf.create_dataset('exponent', data=exponent, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
hf.create_dataset('qual', data=qual, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))

