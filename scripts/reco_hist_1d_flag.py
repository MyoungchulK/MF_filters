import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
#from tools.ara_run_manager import run_info_loader
from tools.ara_known_issue import known_issue_loader
from tools.ara_quality_cut import get_calpulser_cut

Station = int(sys.argv[1])
count_i = int(sys.argv[2])
count_f = int(sys.argv[3])
count_ff = count_i + count_f

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

known_issue = known_issue_loader(Station, verbose = True)
bad_runs = known_issue.get_knwon_bad_run(use_qual = True)
del known_issue

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/reco/*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/qual_cut_full/'
b_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr/'
del d_run_range

runs = np.copy(d_run_tot)
b_runs = np.in1d(runs, bad_runs).astype(int)

pol_name = ['Vpol', 'Hpol']
rs_type = ['41m D', '41m R', '300m D', '300m R']

for r in tqdm(range(len(d_run_tot))):
    
  #if r <10:
  if r >= count_i and r < count_ff:

    if b_runs[r]: continue

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError: 
        print(d_list[r])
        continue
    config = hf['config'][2]
    coord = hf['coord'][:] # pol, thephi, rad, sol, evt
    coef = hf['coef'][:] # pol, rad, sol, evt
    evt = hf['evt_num'][:]
    del hf

    b_name = f'{b_path}snr_A{Station}_R{d_run_tot[r]}.h5'
    hf_b = h5py.File(b_name, 'r')
    trig = hf_b['trig_type'][:]
    rf_t = trig != 0
    del b_name, hf_b, trig

    q_name = f'{q_path}qual_cut_full_A{Station}_R{d_run_tot[r]}.h5'
    hf_q = h5py.File(q_name, 'r')
    evt_full = hf_q['evt_num'][:]
    qual = hf_q['tot_qual_cut_sum'][:] != 0
    cut = np.in1d(evt, evt_full[qual])
    del q_name, hf_q, evt_full, qual

    tot_cut = np.logical_or(rf_t, cut)
    coord[:, :, :, :, tot_cut] = np.nan
    coef[:, :, :, tot_cut] = np.nan
    del cut, rf_t

    cp_cut, num_cuts, pol_idx = get_calpulser_cut(Station, d_run_tot[r])
    cal_cut = np.full((len(evt)), False, dtype = bool)
    for c in range(num_cuts):
        ele_flag = np.digitize(89.5 - coord[pol_idx, 0, 0, 0], cp_cut[c, 0]) == 1
        azi_flag = np.digitize(coord[pol_idx, 1, 0, 0] - 179.5, cp_cut[c, 1]) == 1
        cal_cut += np.logical_and(ele_flag, azi_flag)
        del ele_flag, azi_flag
    coord[:, :, :, :, cal_cut] = np.nan
    coef[:, :, :, cal_cut] = np.nan
    del pol_idx, cp_cut, num_cuts, cal_cut

    scut_val = 35
    zenith_deg = 89.5 - coord[:, 0, 1, :, :] # pol, thetaphi, rad, sol, evt
    zenith_deg = np.reshape(zenith_deg, (4, -1))
    scut = np.any(zenith_deg > scut_val, axis = 0)
    coord[:, :, :, :, scut] = np.nan
    coef[:, :, :, scut] = np.nan
    del scut_val, zenith_deg, scut 

    coef_v = np.nanmax(np.reshape(coef[0], (4, -1)), axis = 0)
    coef_h = np.nanmax(np.reshape(coef[1], (4, -1)), axis = 0)

    entry_v = np.array([-1], dtype = int)
    entry_h = np.array([-1], dtype = int)
    pol_v = -1
    pol_h = -1
    if Station == 2:
        if config == 2 or config == 3 or config == 4:
            if np.any(coef_v > 0.13):
                entry_v = np.where(coef_v > 0.13)[0]
                pol_v = 0
        if config == 6:
            if np.any(coef_h > 0.13):
                entry_h = np.where(coef_h > 0.13)[0]
                pol_h= 1
    if Station == 3:
        if config == 2 or config == 3:
            if np.any(coef_v > 0.14):
                entry_v = np.where(coef_v > 0.14)[0]
                pol_v = 0
        if config == 3 or config == 5:
            if np.any(coef_h > 0.16):
                entry_h = np.where(coef_h > 0.16)[0]   
                pol_h = 1     

    if np.any(entry_v != -1):
        for e in range(len(entry_v)):
            coef_i = np.nanargmax(np.reshape(coef[0], (4, -1))[:, entry_v[e]], axis = 0)
            coord_re = np.reshape(coord[0], (2, 4, -1))[:, coef_i, entry_v[e]]
            print('st:', Station, 'run:', d_run_tot[r], 'config:', config, 'Vpol', 'type:', rs_type[coef_i])
            print('entry:', entry_v[e], 'event:', evt[entry_v[e]])
            print('max corr:', coef_v[entry_v[e]], 'theta:', 89.5 - coord_re[0], 'phi:', coord_re[1] - 179.5)
    if np.any(entry_h != -1):
        for e in range(len(entry_h)):
            coef_i = np.nanargmax(np.reshape(coef[1], (4, -1))[:, entry_h[e]], axis = 0)
            coord_re = np.reshape(coord[1], (2, 4, -1))[:, coef_i, entry_h[e]]
            print('st:', Station, 'run:', d_run_tot[r], 'config:', config, 'Hpol', 'type:', rs_type[coef_i])
            print('entry:', entry_h[e], 'event:', evt[entry_h[e]])
            print('max corr:', coef_h[entry_h[e]], 'theta:', 89.5 - coord_re[0], 'phi:', coord_re[1] - 179.5)

print('done')





