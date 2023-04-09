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
key = str(sys.argv[2])

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/sub_info_sim_{key}/*signal*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

num_flas = 3
num_evts = 2
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

pnu = np.full((d_len, num_evts), np.nan, dtype = float)
cos_angle = np.copy(pnu)
probability = np.copy(pnu)
inu_thrown = np.full((d_len), np.nan, dtype = float)
radius = np.copy(inu_thrown)
sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
flavor = np.copy(sim_run)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    flavor[r] = cons[4]
    radius[r] = hf['radius'][:]
    #inu_thrown[r] = hf['inu_thrown'][-1] + 1
    inu_thrown[r] = hf['nnu_tot'][0]
    pnu[r] = hf['pnu'][:]   
    probability[r] = hf['probability'][:]
    cos_angle[r] = hf['nnu'][3]
    del hf, cons

pnu /= 1e9
cos_angle = np.cos(cos_angle)

solid_angle = 4 * np.pi
area = np.pi * (radius**2)
log_emax0 = 1e12 # 12 GeV
log_emax = 10**(8) # 12 GeV
log_emin = 1e7 # 7 Gev
#one_weight = probability * pnu * area[:, np.newaxis] * solid_angle * (np.log(log_emax) - np.log(log_emin))
one_weight = np.full(pnu.shape, np.nan, dtype = float)
idx1 = np.logical_and(flavor == 1, config == 3)
idx2 = np.logical_and(flavor == 2, config == 3)
print(np.count_nonzero(idx1), np.count_nonzero(idx2))
one_weight[idx1] = probability[idx1] * pnu[idx1] * area[idx1, np.newaxis] * solid_angle * (np.log(10**8) - np.log(10**7))
one_weight[idx2] = probability[idx2] * pnu[idx2] * area[idx2, np.newaxis] * solid_angle * (np.log(10**8) - np.log(10**7))
print(np.log10(np.nanmax(pnu)))
print(np.log10(np.nanmax(pnu[idx1])))
print(np.log10(np.nanmax(pnu[idx2])))

energy_bins = np.logspace(np.log10(log_emin), np.log10(log_emax0), 40 + 1)
cos_bins = np.linspace(-1, 1, 100 + 1)

aeff_1d = np.full((len(energy_bins) - 1, num_flas, num_configs), 0, dtype = float)
aeff_2d = np.full((len(energy_bins) - 1, len(cos_bins) - 1, num_flas, num_configs), 0, dtype = float)
inu_thrown_tot = np.full((num_flas, num_configs), 0, dtype = float)

for f in range(num_flas):
    for c in range(num_configs):
        idxs = np.logical_and(flavor == int(f + 1), config == int(c + 1))
        tot_evt = np.nansum(inu_thrown[idxs])
        tot_pnu = pnu[idxs].flatten()
        tot_cos = cos_angle[idxs].flatten()
        tot_wei = one_weight[idxs].flatten()
        inu_thrown_tot[f, c] = tot_evt

        aeff_1d[:, f, c] = np.histogram(tot_pnu, weights = tot_wei, bins = energy_bins)[0]
        aeff_1d[:, f, c] /= tot_evt * np.diff(energy_bins) * solid_angle

        aeff_2d[:, :, f, c] = np.histogram2d(tot_pnu, tot_cos, weights = tot_wei, bins=(energy_bins, cos_bins))[0]
        aeff_2d[:, :, f, c] /= tot_evt * np.diff(energy_bins)[:, np.newaxis] * np.diff(cos_bins)[np.newaxis, :] * solid_angle

flux_model = np.loadtxt('/home/mkim/analysis/MF_filters/data/flux_data/gzkKoteraSFR1.txt')
energy = flux_model[:,0]
nu_tot_model = flux_model[:,1:]

evt_rate = np.full((pnu.shape), 0, dtype = float)
for f in range(num_flas):
    model_f = interp1d(energy, nu_tot_model[:,f], fill_value="extrapolate")
    fla_idx = flavor == int(f + 1)
    nu_model_int = model_f(np.log10(pnu[fla_idx])) / pnu[fla_idx]
    evt_rate[fla_idx] = one_weight[fla_idx] * nu_model_int
    for c in range(num_configs):
        idxs = np.logical_and(fla_idx, config == int(c + 1))  
        evt_rate[idxs] /= inu_thrown_tot[f, c]

path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'One_Weight_Pad_{key}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
hf.create_dataset('cos_angle', data=cos_angle, compression="gzip", compression_opts=9)
hf.create_dataset('probability', data=probability, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown_tot', data=inu_thrown_tot, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight', data=one_weight, compression="gzip", compression_opts=9)
hf.create_dataset('energy_bins', data=energy_bins, compression="gzip", compression_opts=9)
hf.create_dataset('cos_bins', data=cos_bins, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_1d', data=aeff_1d, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_2d', data=aeff_2d, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('gzkKoteraSFR_energy', data=energy, compression="gzip", compression_opts=9)
hf.create_dataset('gzkKoteraSFR_e2', data=nu_tot_model, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




