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
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*signal*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

num_flas = 3
num_evts = 100
if Station == 2: num_configs = 7
if Station == 3: num_configs = 9

nfour = float(2048 / 2 / 2 * 0.5)

pnu = np.full((d_len, num_evts), np.nan, dtype = float)
cos_angle = np.copy(pnu)
probability = np.copy(pnu)
inu_thrown = np.full((d_len), np.nan, dtype = float)
radius = np.copy(inu_thrown)
sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
flavor = np.copy(sim_run)
exponent = np.full((d_len, 2), 0, dtype = int)
sig_in = np.full((d_len, num_evts), 0, dtype = int)
sig_in_wide = np.full((d_len, num_evts), 0, dtype = int)

for r in tqdm(range(len(d_run_tot))):
    
  #if r < 10:

    hf = h5py.File(d_list[r], 'r')
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    #flavor[r] = cons[4]
    flavor[r] = hf['nuflavorint'][0]
    radius[r] = hf['radius'][:]
    #inu_thrown[r] = hf['inu_thrown'][-1] + 1
    inu_thrown[r] = hf['nnu_tot'][0]
    pnu[r] = hf['pnu'][:]   
    prob = hf['probability'][:]
    prob[prob >= 1] = 0
    probability[r] = prob
    cos_angle[r] = hf['nnu'][3]
    exponent[r] = hf['exponent_range'][:]
    wf_time = hf['wf_time'][:]
    wf_dege = np.array([wf_time[0] - 0.5, wf_time[-1] + 0.5])
    wf_dege_wide = np.array([wf_time[0] - nfour - 0.5, wf_time[-1] + nfour + 0.5])
    sig_bin = hf['signal_bin'][:]
    sig_in[r] = np.nansum(np.digitize(sig_bin, wf_dege) == 1, axis = (0, 1))
    sig_in_wide[r] = np.nansum(np.digitize(sig_bin, wf_dege_wide) == 1, axis = (0, 1))
    del hf, cons, prob

print(np.count_nonzero(sig_in == 0))
print(np.count_nonzero(sig_in_wide == 0))
print(np.nansum(inu_thrown))
print(np.count_nonzero(sig_in == 0) / np.nansum(inu_thrown))
print(np.count_nonzero(sig_in_wide == 0) / np.nansum(inu_thrown))

pnu /= 1e9
exponent -= 9
cos_angle = np.cos(cos_angle)
exponent_lin = 10 ** (exponent)
solid_angle = 4 * np.pi
area = np.pi * (radius**2)
one_weight = probability * pnu * area[:, np.newaxis] * solid_angle * (np.log(exponent_lin[:, 1][:, np.newaxis]) - np.log(exponent_lin[:, 0][:, np.newaxis]))
one_weight_sig = np.copy(one_weight)
one_weight_sig[sig_in == 0] = 0
one_weight_sig_wide = np.copy(one_weight)
one_weight_sig_wide[sig_in_wide == 0] = 0

ex_range = np.arange(7, 13, 1, dtype = int)
num_ens = len(ex_range)
energy_bins = np.logspace(ex_range[0], ex_range[-1], 40 + 1)
cos_bins = np.linspace(-1, 1, 100 + 1)

aeff_1d = np.full((len(energy_bins) - 1, num_flas, num_configs, num_ens), 0, dtype = float)
aeff_2d = np.full((len(energy_bins) - 1, len(cos_bins) - 1, num_flas, num_configs, num_ens), 0, dtype = float)
aeff_1d_sig = np.copy(aeff_1d)
aeff_2d_sig = np.copy(aeff_2d)
aeff_1d_sig_wide = np.copy(aeff_1d)
aeff_2d_sig_wide = np.copy(aeff_2d)
inu_thrown_tot = np.full((num_flas, num_configs, num_ens), 0, dtype = float)

for f in range(num_flas):
    for c in range(num_configs):
        for e in range(num_ens):
            idxs = np.all((flavor == int(f + 1), config == int(c + 1), exponent[:, 0] == ex_range[e]), axis = 0)
            tot_evt = np.nansum(inu_thrown[idxs])
            tot_pnu = pnu[idxs].flatten()
            tot_cos = cos_angle[idxs].flatten()
            tot_wei = one_weight[idxs].flatten()
            tot_wei_sig = one_weight_sig[idxs].flatten()
            tot_wei_sig_wide = one_weight_sig_wide[idxs].flatten()
            inu_thrown_tot[f, c, e] = tot_evt

            aeff_1d[:, f, c, e] = np.histogram(tot_pnu, weights = tot_wei, bins = energy_bins)[0]
            aeff_1d[:, f, c, e] /= tot_evt * np.diff(energy_bins) * solid_angle
            aeff_1d_sig[:, f, c, e] = np.histogram(tot_pnu, weights = tot_wei_sig, bins = energy_bins)[0]
            aeff_1d_sig[:, f, c, e] /= tot_evt * np.diff(energy_bins) * solid_angle
            aeff_1d_sig_wide[:, f, c, e] = np.histogram(tot_pnu, weights = tot_wei_sig_wide, bins = energy_bins)[0]
            aeff_1d_sig_wide[:, f, c, e] /= tot_evt * np.diff(energy_bins) * solid_angle

            aeff_2d[:, :, f, c, e] = np.histogram2d(tot_pnu, tot_cos, weights = tot_wei, bins=(energy_bins, cos_bins))[0]
            aeff_2d[:, :, f, c, e] /= tot_evt * np.diff(energy_bins)[:, np.newaxis] * np.diff(cos_bins)[np.newaxis, :] * solid_angle
            aeff_2d_sig[:, :, f, c, e] = np.histogram2d(tot_pnu, tot_cos, weights = tot_wei_sig, bins=(energy_bins, cos_bins))[0]
            aeff_2d_sig[:, :, f, c, e] /= tot_evt * np.diff(energy_bins)[:, np.newaxis] * np.diff(cos_bins)[np.newaxis, :] * solid_angle
            aeff_2d_sig_wide[:, :, f, c, e] = np.histogram2d(tot_pnu, tot_cos, weights = tot_wei_sig_wide, bins=(energy_bins, cos_bins))[0]
            aeff_2d_sig_wide[:, :, f, c, e] /= tot_evt * np.diff(energy_bins)[:, np.newaxis] * np.diff(cos_bins)[np.newaxis, :] * solid_angle

m_to_cm = 1e4
flux_model = np.loadtxt('/home/mkim/analysis/MF_filters/data/flux_data/gzkKoteraSFR1.txt')
energy = flux_model[:,0]
nu_tot_model = flux_model[:,1:] * m_to_cm

evt_rate = np.full((pnu.shape), 0, dtype = float)
for f in range(num_flas):
    model_f = interp1d(energy, nu_tot_model[:,f], fill_value="extrapolate")
    fla_idx = flavor == int(f + 1)
    nu_model_int = model_f(np.log10(pnu[fla_idx])) / pnu[fla_idx]
    evt_rate[fla_idx] = one_weight[fla_idx] * nu_model_int
    for c in range(num_configs):
        for e in range(num_ens):
            idxs = np.all((fla_idx, config == int(c + 1), exponent[:, 0] == ex_range[e]), axis = 0)
            evt_rate[idxs] /= inu_thrown_tot[f, c, e]
evt_rate_sig = np.copy(evt_rate)
evt_rate_sig[sig_in == 0] = 0
evt_rate_sig_wide = np.copy(evt_rate)
evt_rate_sig_wide[sig_in_wide == 0] = 0

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'One_Weight_Pad_{key}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
hf.create_dataset('exponent', data=exponent, compression="gzip", compression_opts=9)
hf.create_dataset('cos_angle', data=cos_angle, compression="gzip", compression_opts=9)
hf.create_dataset('probability', data=probability, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown_tot', data=inu_thrown_tot, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight', data=one_weight, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight_sig', data=one_weight_sig, compression="gzip", compression_opts=9)
hf.create_dataset('one_weight_sig_wide', data=one_weight_sig_wide, compression="gzip", compression_opts=9)
hf.create_dataset('energy_bins', data=energy_bins, compression="gzip", compression_opts=9)
hf.create_dataset('cos_bins', data=cos_bins, compression="gzip", compression_opts=9)
hf.create_dataset('sig_in', data=sig_in, compression="gzip", compression_opts=9)
hf.create_dataset('sig_in_wide', data=sig_in_wide, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_1d', data=aeff_1d, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_1d_sig', data=aeff_1d_sig, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_1d_sig_wide', data=aeff_1d_sig_wide, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_2d', data=aeff_2d, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_2d_sig', data=aeff_2d_sig, compression="gzip", compression_opts=9)
hf.create_dataset('aeff_2d_sig_wide', data=aeff_2d_sig_wide, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_sig', data=evt_rate_sig, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate_sig_wide', data=evt_rate_sig_wide, compression="gzip", compression_opts=9)
hf.create_dataset('gzkKoteraSFR_energy', data=energy, compression="gzip", compression_opts=9)
hf.create_dataset('gzkKoteraSFR_e2', data=nu_tot_model, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




