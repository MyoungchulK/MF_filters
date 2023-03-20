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

def get_calpulser_cut(st, run):

    if st == 2:
        cp6 = np.full((2, 2), np.nan, dtype = float)
        cp6[0, 0] = -1.45       
        cp6[0, 1] = 10.45       
        cp6[1, 0] = 56.65       
        cp6[1, 1] = 69.15
        cp5 = np.full((2, 2), np.nan, dtype = float)
        cp5[0, 0] = -28
        cp5[0, 1] = -19
        cp5[1, 0] = -29.35
        cp5[1, 1] = -21.75
        cp5_m = np.full((2, 2), np.nan, dtype = float)
        cp5_m[0, 0] = 27.15
        cp5_m[0, 1] = 37.75
        cp5_m[1, 0] = -29.35
        cp5_m[1, 1] = -21.75
        cp5_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_2020[0, 0] = -27.85
        cp5_2020[0, 1] = -20.25
        cp5_2020[1, 0] = -31.05
        cp5_2020[1, 1] = -21.65
        cp5_m_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_m_2020[0, 0] = 28.65
        cp5_m_2020[0, 1] = 37.15
        cp5_m_2020[1, 0] = -45.85
        cp5_m_2020[1, 1] = -38.35       

        if run < 1901:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 1900 and run < 1935:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
        elif run == 1935:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
            cp_cut[1] = cp6
        elif run > 1935 and run < 7006:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 7005 and run < 8098:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
            cp_cut[1] = cp5_m
        elif run > 8097 and run < 9505:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 9504 and run < 15527:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5
        elif run > 15526:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2020
            cp_cut[1] = cp5_m_2020
        num_cuts = cp_cut.shape[0] 

    elif st == 3:
        cp6 = np.full((2, 2), np.nan, dtype = float)
        cp6[0, 0] = -16.75
        cp6[0, 1] = -12.25
        cp6[1, 0] = 61.25
        cp6[1, 1] = 65.75
        cp5_2020 = np.full((2, 2), np.nan, dtype = float)
        cp5_2020[0, 0] = -18.25
        cp5_2020[0, 1] = -12.65
        cp5_2020[1, 0] = -27.65
        cp5_2020[1, 1] = -18.45
        cp5_2019 = np.full((2, 2), np.nan, dtype = float)
        cp5_2019[0, 0] = -18.25
        cp5_2019[0, 1] = -12.65
        cp5_2019[1, 0] = -35.05
        cp5_2019[1, 1] = 1.75
        cp6_m_2019 = np.full((2, 2), np.nan, dtype = float)
        cp6_m_2019[0, 0] = -21.25
        cp6_m_2019[0, 1] = -16.35
        cp6_m_2019[1, 0] = -117.75
        cp6_m_2019[1, 1] = -113.95

        if run < 12873:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
        elif run > 12872 and run < 13901:
            cp_cut = np.full((2, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp6
            cp_cut[1] = cp6_m_2019
        elif run > 13900 and run < 16487:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2019
        elif run > 16486:
            cp_cut = np.full((1, 2, 2), np.nan, dtype = float)
            cp_cut[0] = cp5_2020
        num_cuts = cp_cut.shape[0]

    return cp_cut, num_cuts

def get_calpulser_pol(st, run):

    pol_idx = 0
    if st == 2 and (run > 1877 and run < 1887):
        pol_idx = 1
    elif st == 3 and (run > 923 and run < 934):
        pol_idx = 1
    
    return pol_idx

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

    pol_idx = get_calpulser_pol(Station, d_run_tot[r])
    cp_cut, num_cuts = get_calpulser_cut(Station, d_run_tot[r])
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

    entry = -1
    pol = -1
    if Station == 2:
        if config == 2 or config == 3 or config == 4:
            if np.any(coef_v > 0.13):
                entry = np.where(coef_v > 0.13)[0]
                pol = 0
        if config == 6:
            if np.any(coef_h > 0.13):
                entry = np.where(coef_h > 0.13)[0]
                pol = 1
    if Station == 3:
        if config == 2 or config == 3:
            if np.any(coef_v > 0.14):
                entry = np.where(coef_v > 0.14)[0]
                pol = 0
        if config == 3 or config == 5:
            if np.any(coef_h > 0.16):
                entry = np.where(coef_h > 0.16)[0]   
                pol = 1     

    if entry == -1: continue
    coefs = [coef_v, coef_h]
    for e in range(len(entry)):
        coef_i = np.nanargmax(np.reshape(coef[pol], (4, -1))[:, entry[e]], axis = 0)
        coord_re = np.reshape(coord[pol], (2, 4, -1))[:, coef_i, entry[e]]
        print('st:', Station, 'run:', d_run_tot[r], 'config:', config, 'pol:', pol_name[pol], 'type:', rs_type[coef_i])
        print('entry:', entry[e], 'event:', evt[entry[e]])
        print('max corr:', coefs[pol][entry[e]], 'theta:', 89.5 - coord_re[0], 'phi:', coord_re[1] - 179.5)

print('done')





