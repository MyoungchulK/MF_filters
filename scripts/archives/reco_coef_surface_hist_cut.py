import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
print(bad_runs)
print(f'# of bad runs: {len(bad_runs)}')

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range = file_sorter(d_path)
d_len = len(d_run_tot)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'

c_bins = np.linspace(0, 1.2, 1200 + 1, dtype = float)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
c_bin_len = len(c_bin_center)
z_bins = np.linspace(0, 180, 180 + 1, dtype = int)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
z_bin_len = len(z_bin_center)

coef_tot = np.full((c_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # c, trig, pol, rad, sol, config
coef_cut = np.copy(coef_tot)

zenith_tot = np.full((z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # z, trig, pol, rad, sol, config
zenith_cut = np.copy(zenith_tot)

cz_tot = np.full((c_bin_len, z_bin_len, 3, 2, 2, 2, num_configs), 0, dtype = int) # c, z, trig, pol, rad, sol, config
cz_cut = np.copy(cz_tot)

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:

    bad_idx = d_run_tot[r] in bad_runs
    #    print('bad run:', d_list[r], d_run_tot[r])
    #    continue

    ara_run = run_info_loader(Station, d_run_tot[r])
    g_idx = ara_run.get_config_number() - 1
    del ara_run

    hf = h5py.File(d_list[r], 'r')
    trig = hf['trig_type'][:]
    rf_t = trig == 0
    cal_t = trig == 1
    soft_t = trig == 2
    t_list = [rf_t, cal_t, soft_t]
    coef = hf['coef'][:] # pol, rad, sol, evt
    coord = hf['coord'][:, 0] # pol, rad, sol, evt
    evt = hf['evt_num'][:]
    del hf, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:]
    cut = np.in1d(evt, evt_full[qual != 0])
    del q_name, hf_q, qual, evt_full

    coef_c = np.copy(coef)
    coef_c[:,:,:,cut] = np.nan
    coord_c = np.copy(coord)
    coord_c[:,:,:,cut] = np.nan
    del cut

    for t in range(3):
        for pol in range(2):
            for rad in range(2):
                for sol in range(2):       
                    coef_tot[:, t, pol, rad, sol, g_idx] += np.histogram(coef[pol, rad, sol][t_list[t]], bins = (c_bins))[0].astype(int)
                    zenith_tot[:, t, pol, rad, sol, g_idx] += np.histogram(coord[pol, rad, sol][t_list[t]], bins = (z_bins))[0].astype(int)
                    cz_tot[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coef[pol, rad, sol][t_list[t]], coord[pol, rad, sol][t_list[t]], bins = (c_bins, z_bins))[0].astype(int)
                    if bad_idx:
                        continue
                    coef_cut[:, t, pol, rad, sol, g_idx] += np.histogram(coef_c[pol, rad, sol][t_list[t]], bins = (c_bins))[0].astype(int)
                    zenith_cut[:, t, pol, rad, sol, g_idx] += np.histogram(coord_c[pol, rad, sol][t_list[t]], bins = (z_bins))[0].astype(int)
                    cz_cut[:, :, t, pol, rad, sol, g_idx] += np.histogram2d(coef_c[pol, rad, sol][t_list[t]], coord_c[pol, rad, sol][t_list[t]], bins = (c_bins, z_bins))[0].astype(int)
    del coef, coef_c, t_list, rf_t, cal_t, soft_t, coord_c, coord

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Reco_Coef_Zenith_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('coef_tot', data=coef_tot, compression="gzip", compression_opts=9)
hf.create_dataset('coef_cut', data=coef_cut, compression="gzip", compression_opts=9)
hf.create_dataset('zenith_tot', data=zenith_tot, compression="gzip", compression_opts=9)
hf.create_dataset('zenith_cut', data=zenith_cut, compression="gzip", compression_opts=9)
hf.create_dataset('cz_tot', data=cz_tot, compression="gzip", compression_opts=9)
hf.create_dataset('cz_cut', data=cz_cut, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)






