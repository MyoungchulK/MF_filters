import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm
from scipy.interpolate import interp1d

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
if Station == 2:
    config_len = 6
if Station == 3:
    config_len = 7

i_key = '_C'
i_key_len = len(i_key)
f_key = '_E1'
fi_key = '_Nu'
fi_key_len = len(fi_key)
ff_key = '_signal'
ri_key = 'run'
ri_key_len = len(ri_key)
rf_key = '.h5'

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/mf_sim/'
d_list, d_run_tot, d_run_range = file_sorter(d_path+'*signal*')

evt_len = 100 
run_len = 100 
flavor_len = 3

radius = np.array([1000000])
pnu = np.full((flavor_len, config_len, run_len, evt_len), np.nan, dtype = float)
prob = np.copy(pnu)
nnu = np.copy(pnu)
thrown = np.full((flavor_len, config_len, run_len), np.nan, dtype = float)

for r in tqdm(range(len(d_run_tot))):
    file_name = d_list[r]
    i_idx = file_name.find(i_key)
    f_idx = file_name.find(f_key, i_idx + i_key_len)
    c_idx = int(file_name[i_idx + i_key_len:f_idx]) - 1

    fi_idx = file_name.find(fi_key)
    ff_idx = file_name.find(ff_key, fi_idx + fi_key_len)
    fla = file_name[fi_idx + fi_key_len:ff_idx]
    if fla == 'E': fla_idx = 0
    if fla == 'Mu': fla_idx = 1
    if fla == 'Tau': fla_idx = 2

    ri_idx = file_name.find(ri_key)
    rf_idx = file_name.find(rf_key, ri_idx + ri_key_len)
    r_idx = int(file_name[ri_idx + ri_key_len:rf_idx])

    hf = h5py.File(d_list[r], 'r')
    nnu_r = np.cos(hf['nnu'][3])
    pnu_r = hf['pnu'][:]/1e9
    prob_r = hf['probability'][:]
    thrown_r = hf['inu_thrown'][-1]

    nnu[fla_idx, c_idx, r_idx, :] = nnu_r
    pnu[fla_idx, c_idx, r_idx, :] = pnu_r
    prob[fla_idx, c_idx, r_idx, :] = prob_r
    thrown[fla_idx, c_idx, r_idx] = thrown_r
    del hf, file_name, i_idx, f_idx, c_idx, fi_idx, ff_idx, fla, fla_idx
    del nnu_r, pnu_r, prob_r, thrown_r, ri_idx, rf_idx, r_idx

area = np.pi*((radius)**2)    
e_range = np.array([7,12])
one_weight = prob * pnu * (np.log(np.power(10, e_range[1])) - np.log(np.power(10, e_range[0]))) * (4*np.pi) * area

p_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/Reco_MF_Map_W_Sim_A{Station}.h5'
hf_p = h5py.File(p_path, 'r')
evt_rate_pass = hf_p['evt_rate_pass'][:]
one_weight[np.isnan(evt_rate_pass)] = 0

thrown_sum = np.nansum(thrown, axis = 2)
e_bins = np.logspace(7, 12, 40+1)
e_bin_center = (e_bins[1:] + e_bins[:-1]) / 2
t_bins = np.linspace(-1, 1, 100+1)
t_bin_center = (t_bins[1:] + t_bins[:-1]) / 2
aeff = np.full((flavor_len, config_len, len(e_bin_center)), np.nan, dtype = float)
aeff_2d = np.full((flavor_len, config_len, len(e_bin_center), len(t_bin_center)), np.nan, dtype = float)
for f in range(flavor_len):
    for c in range(config_len):
        aeff[f,c] = np.histogram(pnu[f,c].flatten(), weights=one_weight[f,c].flatten(), bins=e_bins)[0]
        aeff[f,c] /= thrown_sum[f,c] * np.diff(e_bins) * (4*np.pi)

        aeff_2d[f,c] = np.histogram2d(pnu[f,c].flatten(), nnu[f,c].flatten(), weights=one_weight[f,c].flatten(), bins=(e_bins, t_bins))[0]
        aeff_2d[f,c] /= thrown_sum[f,c] * np.diff(e_bins)[:, np.newaxis] * np.diff(t_bins)[np.newaxis, :] * (4*np.pi)

if Station == 2: tot_live_time = np.array([4.8])
if Station == 3: tot_live_time = np.array([4.6])
flux_model = np.loadtxt('/home/mkim/analysis/MF_filters/data/flux_data/gzkKoteraSFR1.txt')
energy = flux_model[:,0]
nu_tot_model = flux_model[:,1:]

evt_rate = np.full((flavor_len, config_len, run_len, evt_len), np.nan, dtype = float)
evt_rate_livetime = np.copy(evt_rate)
for f in range(flavor_len):
    model_f = interp1d(energy, nu_tot_model[:,f], fill_value="extrapolate")
    for c in range(config_len):
        for r in range(run_len):
            nu_model_int = model_f(np.log10(pnu[f,c,r])) / pnu[f,c,r]
            evt_rate[f,c,r] = one_weight[f,c,r] * nu_model_int / thrown_sum[f,c]
            evt_rate_livetime[f,c,r] = evt_rate[f,c,r] * 60 * 60 * 24 * 365 * tot_live_time

evt_rate_flavor = np.nansum(evt_rate_livetime, axis = (2,3))
print(evt_rate_flavor)
print(np.round(evt_rate_flavor,2))

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)
file_name = f'Weight_Sim_Pass_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('area', data=area, compression="gzip", compression_opts=9)
hf.create_dataset('e_range', data=e_range, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
hf.create_dataset('prob', data=prob, compression="gzip", compression_opts=9)
hf.create_dataset('nnu', data=nnu, compression="gzip", compression_opts=9)
hf.create_dataset('thrown', data=thrown, compression="gzip", compression_opts=9)
hf.create_dataset('thrown_sum', data=thrown_sum, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight', data=one_weight, compression="gzip", compression_opts=9)
hf.create_dataset('e_bins', data=e_bins, compression="gzip", compression_opts=9)
hf.create_dataset('t_bins', data=t_bins, compression="gzip", compression_opts=9)
hf.create_dataset('e_bin_center', data=e_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('t_bin_center', data=t_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('aeff', data=aeff, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_2d', data=aeff_2d, compression="gzip", compression_opts=9)
hf.create_dataset('tot_live_time', data=tot_live_time, compression="gzip", compression_opts=9)
hf.create_dataset('energy', data=energy, compression="gzip", compression_opts=9)
hf.create_dataset('nu_tot_model', data=nu_tot_model, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_livetime', data=evt_rate_livetime, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_flavor', data=evt_rate_flavor, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name)
# quick size check
size_checker(path+file_name)
