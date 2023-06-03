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
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_sim/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

sig_noi = np.full((d_len), 0, dtype = int)
configs = np.full((d_len), 0, dtype = int)

p_bins = np.linspace(0, 4000, 400 + 1)
p_bin_center = (p_bins[1:] + p_bins[:-1]) / 2
p_bin_len = len(p_bin_center)

p2p_all = np.full((d_len, p_bin_len, 16, 2), 0, dtype = int)
print(p2p_all.shape)
del p_bin_len, d_len

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    cons = hf['config'][:]
    configs[r] = cons[2]
    sig_noi[r] = int(cons[4] == -1)
    snr = hf['p2p'][:] # (num_ants, num_evts)
    del hf, cons

    q_name = d_list[r].replace('rms', 'qual_cut')
    hf_q = h5py.File(q_name, 'r')
    cut = hf_q['tot_qual_cut_sum'][:] != 0
    del q_name, hf_q

    ex_run = get_example_run(Station, configs[r])
    bad_ant = known_issue.get_bad_antenna(ex_run)
    snr_cut = np.copy(snr)
    snr_cut[:, cut] = np.nan
    snr_cut[bad_ant] = np.nan
    del cut, ex_run, bad_ant

    for a in range(16):
        p2p_all[r, :, a, 0] = np.histogram(snr[a], bins = p_bins)[0].astype(int)
        p2p_all[r, :, a, 1] = np.histogram(snr_cut[a], bins = p_bins)[0].astype(int)
    del snr, snr_cut

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'P2P_Sim_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('p_bins', data=p_bins, compression="gzip", compression_opts=9)
hf.create_dataset('p_bin_center', data=p_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sig_noi', data=sig_noi, compression="gzip", compression_opts=9)
hf.create_dataset('configs', data=configs, compression="gzip", compression_opts=9)
hf.create_dataset('p2p_all', data=p2p_all, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






