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
from tools.ara_run_manager import get_path_info_v2

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/rms_sim/*signal*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range, d_path 

q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/One_Weight_Pad_mass_A{Station}.h5'
hf = h5py.File(q_path, 'r')
prob = hf['probability'][:]
evt_rate = hf['evt_rate'][:]
evt_rate[prob >= 1] = 0
print(prob.shape)
print(evt_rate.shape)
fla_w = hf['flavor'][:]
en_w = hf['exponent'][:, 0]
con_w = hf['config'][:]
run_w = hf['sim_run'][:]
print(fla_w.shape)
print(en_w.shape)
print(con_w.shape)
print(run_w.shape)
del q_path, hf, prob

r_bins = np.linspace(0, 30, 300 + 1)
r_bin_center = (r_bins[1:] + r_bins[:-1]) / 2
r_bin_len = len(r_bin_center)

pow_r = np.full((d_len, r_bin_len), 0, dtype = int)
pow_r_w = np.full((d_len, r_bin_len), 0, dtype = float)
del r_bin_len

config = np.full((d_len), 0, dtype = int)
sim_run = np.copy(config)
flavor = np.copy(config)
en = np.copy(config)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    con = hf['config'][:]
    sim_run[r] = int(con[1])
    config[r] = int(con[2])
    flavor[r] = int(con[4])
    en[r] = int(get_path_info_v2(d_list[r], '_E', '_F')) - 9
    w_idx = np.all((fla_w == flavor[r], en_w == en[r], con_w == config[r], run_w == sim_run[r]), axis = 0)
    evt_w = evt_rate[w_idx][0]
    rms = hf['rms'][:]
    del hf, con, w_idx

    ex_run = get_example_run(Station, config[r])
    bad_ant = known_issue.get_bad_antenna(ex_run)
    rms[bad_ant] = np.nan
    pow_n = rms ** 2
    pow_n_avg = np.full((4, 100), np.nan, dtype = float)
    for m in range(4):
        pow_n_avg[m] = np.nanmean(pow_n[m::4], axis = 0)
    pow_n_avg_sort = -np.sort(-pow_n_avg, axis = 0)    
    pow_ratio = pow_n_avg_sort[0] / pow_n_avg_sort[1]
    del pow_n_avg_sort, pow_n_avg, pow_n, rms, bad_ant, ex_run

    pow_r[r] = np.histogram(pow_ratio, bins = r_bins)[0].astype(int)    
    pow_r_w[r] = np.histogram(pow_ratio, bins = r_bins, weights = evt_w)[0] 
    del pow_ratio, evt_w

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'PowRatio_Sim_1d_full_A{Station}_R{count_i}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('r_bins', data=r_bins, compression="gzip", compression_opts=9)
hf.create_dataset('r_bin_center', data=r_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('en', data=en, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r', data=pow_r, compression="gzip", compression_opts=9)
hf.create_dataset('pow_r_w', data=pow_r_w, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))






