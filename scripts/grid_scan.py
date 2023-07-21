import os, sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import math

def log_linear_fit(x, a, b):
    y = np.exp(a*x + b)
    return y

def log_linear_fit_inter(x, a, b):
    y = -np.exp(a*x + b) / a
    return y

def lnGamma(z, x, n=100):
    if z > 0 and x > 0 and n > 2:
        if x > 100:
            step1_1 = x / 100
            mult1_1 = step1_1 * math.log(math.exp(-100))
        else:
            mult1_1 = math.log(math.exp(-x))

        if z > 10:
            step = z / 10
            mult1_2 = step * math.log(math.pow(x, 10))
        else:
            mult1_2 = math.log(math.pow(x, z))

        mult2 = 0

        for i in range(n, 0, -1):
            mult2 = i * (i - z) / (x + 2 * i + 1 - z - mult2)

        mult2 = 1 / (x + 1 - z - mult2)
        mult2 = math.log(mult2)

        return mult1_1 + mult1_2 + mult2
    else:
        return 0


def Alpha_nb_ln(s_up, nb, n=100):
    if s_up > 0 and nb > 0:
        gamma_part = lnGamma(1 + nb, s_up + nb, n) - lnGamma(1 + nb, nb, n)
        gamma_part = math.exp(gamma_part)
        return 1 - gamma_part
    else:
        return 0


def GetS_up(ExpEvts, alpha_cut, n=100):
    s_up = -0.01
    alpha_out = 0
    while alpha_out < alpha_cut:
        s_up += 0.01
        alpha_out = Alpha_nb_ln(s_up, ExpEvts, n)
    return s_up, alpha_out

Station = int(sys.argv[1])
Pol = str(sys.argv[2])
Config = int(sys.argv[3])

# sort
dpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

if Station == 2:
    num_configs = 7
if Station == 3:
    num_configs = 9

trig_name = ['RF', 'Cal', 'Soft']
trig = 'RF'
trig_idx = trig_name.index(trig)

wfname = f'Data_Summary_v1_A{Station}.h5'
hf_d = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
#for f in list(hf_d):
#    print(f)
config_d = hf_d[f'configs'][:]
print(config_d.shape)
livetime_d = hf_d['livetime'][:,1]
print(livetime_d.shape)
con_ep = hf_d['con_ep'][:]
print(con_ep.shape)
trig_d = hf_d['trig_ep'][:]
print(trig_d.shape)
qual_d = hf_d['qual_ep'][:]
print(qual_d.shape)
good_idx = np.logical_and(trig_d == trig_idx, qual_d == 0)
del trig_d, qual_d

wfname = f'Sim_Summary_signal_v1_A{Station}.h5'
hf_s = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
#for f in (list(hf_s)):
#  print(f)
config_s = hf_s['config'][:]
print(config_s.shape)
flavor_s = hf_s['flavor'][:]
print(flavor_s.shape)
sig_in_wide_s = hf_s['sig_in_wide'][:] == 0
print(sig_in_wide_s.shape)
qual_tot_s = hf_s['qual_tot'][:] != 0
print(qual_tot_s.shape)
qual_s = np.logical_or(sig_in_wide_s, qual_tot_s)
print(qual_s.shape)
evt_rate_s = hf_s['evt_rate'][:]
print(evt_rate_s.shape)
evt_rate_s[qual_s] = np.nan
del sig_in_wide_s, qual_tot_s

pol_name = ['VPol', 'HPol']
pol = f'{Pol}Pol'
pol_idx = pol_name.index(pol)
fla_name = ['NuE', 'NuMu', 'NuTau']

dat_hf_c = 'coef_max'
dat_hf_m = 'mf_max'

dat_d_c = hf_d[dat_hf_c][pol_idx]
dat_d_c[~good_idx] = np.nan
print(dat_d_c.shape)
dat_d_m = hf_d[dat_hf_m][pol_idx]
dat_d_m[~good_idx] = np.nan
print(dat_d_m.shape)
dat_d = [dat_d_c, dat_d_m]
del good_idx

dat_s_c = hf_s[dat_hf_c][:, pol_idx]
dat_s_c[qual_s] = np.nan
print(dat_s_c.shape)
dat_s_m = hf_s[dat_hf_m][:, pol_idx]
dat_s_m[qual_s] = np.nan
print(dat_s_m.shape)
dat_s = [dat_s_c, dat_s_m]
del qual_s

m_range = np.linspace(-500, 0, 250 + 1, dtype = float)
d_range = np.linspace(0, 200, 400 + 1, dtype = float)
print(m_range.shape, d_range.shape)

evt_tot = np.full((3, num_configs), 0, dtype = float)
livesec = np.full((num_configs), 0, dtype = float)

dat_cut_hist = np.full((len(d_range), len(m_range), num_configs), 0, dtype = float)
sim_s_cut_hist_indi = np.full((len(d_range), len(m_range), 3, num_configs), 0, dtype = float)
sim_s_pass_hist_indi = np.copy(sim_s_cut_hist_indi)
for a in range(num_configs):
  if a != int(Config - 1): continue
  xdat_c = dat_d[0][con_ep == int(a + 1)]
  ydat_c = dat_d[1][con_ep == int(a + 1)]
  con_d_idx = config_d == int(a + 1)
  livesec[a] = np.nansum(livetime_d[con_d_idx])

  con_s_idx = config_s == int(a + 1)
  xsim_s = []
  ysim_s = []
  evt_count_s = []
  for f in range(3):
    idxs = np.logical_and(flavor_s == int(f + 1), con_s_idx)
    xsim_ss = dat_s[0][idxs].flatten()
    ysim_ss = dat_s[1][idxs].flatten()
    evt_count_ss = evt_rate_s[idxs].flatten() * livesec[a]
    evt_tot[f, a] = np.nansum(evt_count_ss)
    xsim_s.append(xsim_ss)
    ysim_s.append(ysim_ss)
    evt_count_s.append(evt_count_ss)
    del idxs
  del con_d_idx, con_s_idx

  for d in tqdm(range(len(d_range))):
    for m in range(len(m_range)):
      dat_cut_hist[d, m, a] = np.count_nonzero(ydat_c < (xdat_c * m_range[m] + d_range[d])) * 10
      for f in range(3):
        s_idx = ysim_s[f] < (xsim_s[f] * m_range[m] + d_range[d])
        sim_s_cut_hist_indi[d, m, f, a] = np.nansum(s_idx.astype(float) * evt_count_s[f])
        sim_s_pass_hist_indi[d, m, f, a] = np.nansum((~s_idx).astype(float) * evt_count_s[f])
        del s_idx
  del xdat_c, ydat_c, xsim_s, ysim_s, evt_count_s

sim_s_cut_hist = np.nanmean(sim_s_cut_hist_indi, axis = 2)
sim_s_pass_hist = np.nanmean(sim_s_pass_hist_indi, axis = 2)
live_days = livesec / (60 * 60 * 24)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = path+f'grid_scan_A{Station}_{Pol}_C{Config}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('m_range', data=m_range, compression="gzip", compression_opts=9)
hf.create_dataset('d_range', data=d_range, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('dat_cut_hist', data=dat_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_cut_hist_indi', data=sim_s_cut_hist_indi, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_cut_hist', data=sim_s_cut_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist_indi', data=sim_s_pass_hist_indi, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist', data=sim_s_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.close()

print(file_name)
print('done!')






