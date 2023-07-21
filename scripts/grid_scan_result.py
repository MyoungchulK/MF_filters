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
file_name = f'{dpath}grid_scan_A{Station}_{Pol}_R{Config}.h5'

if Station == 2:
    num_configs = 7
if Station == 3:
    num_configs = 9

hf = h5py.File(file_name, 'r')
print(file_name)
for f in list(hf):
    print(f)

m_range = hf['m_range'][:]
print(m_range.shape)
d_range = hf['d_range'][:]
print(d_range.shape)
evt_tot = hf['evt_tot'][:]
print(evt_tot.shape)
livesec = hf['livesec'][:]
print(livesec.shape)
live_days = hf['live_days'][:]
print(live_days.shape)
dat_cut_hist = hf['dat_cut_hist'][:]
print(dat_cut_hist.shape)
sim_s_pass_hist = hf['sim_s_pass_hist'][:]
print(sim_s_pass_hist.shape)
sim_s_cut_hist = hf['sim_s_cut_hist'][:]
print(sim_s_cut_hist.shape)
d_diff_range = d_range[1:]
dat_cut_diff_hist = np.diff(dat_cut_hist, axis = 0)
print(dat_cut_diff_hist.shape)
sim_s_cut_diff_hist = np.diff(sim_s_cut_hist, axis = 0)
print(sim_s_cut_diff_hist.shape)

pol_name = ['VPol', 'HPol']
pol = 'VPol'
pol_idx = pol_name.index(pol)

trig_name = ['RF', 'Cal', 'Soft']
trig = 'RF'
trig_idx = trig_name.index(trig)

fla_name = ['NuE', 'NuMu', 'NuTau']

fit_line = np.full(dat_cut_diff_hist.shape, np.nan, dtype = float)
fit_line_fill = np.copy(fit_line)
fit_param = np.full((2, len(m_range), num_configs), np.nan, dtype = float)
back_est = np.full(dat_cut_hist.shape, np.nan, dtype = float)

for a in tqdm(range(num_configs)):
  if a == int(Config - 1):
   for m in range(len(m_range)):
    max_idx = np.nanargmax(dat_cut_diff_hist[:, m, a])
    end_idx = np.where(dat_cut_diff_hist[max_idx:, m, a] < 9.5)[0][0] + max_idx
    mid_idx = max_idx + (end_idx - max_idx) // 2
    del max_idx

    fix_x = d_diff_range[mid_idx:end_idx]
    fit_y = dat_cut_diff_hist[mid_idx:end_idx, m, a]
    fix_x_p = d_diff_range[mid_idx:]
    del end_idx

    popt, pcov = curve_fit(log_linear_fit, fix_x, fit_y, bounds = ([-np.inf,1],[np.inf,np.inf]))
    del pcov
    fit_param[:, m, a] = popt
    fit_line_fill[:, m, a] = dat_cut_diff_hist[:, m, a]
    fit_line_fill[mid_idx:, m, a]  = log_linear_fit(fix_x_p, *popt)
    fit_line[mid_idx:, m, a]  = log_linear_fit(fix_x_p, *popt)
    back_est[:, m, a] = log_linear_fit_inter(d_range, *popt)
    del mid_idx, fix_x, fit_y, fix_x_p, popt

#back_est_sum = np.nansum(back_est, axis = -1)
#sim_s_pass_hist_sum  = np.nansum(sim_s_pass_hist, axis = -1)

sup_arr = np.full(back_est.shape, 0, dtype = float)
#sup_arr_sum = np.full((back_est.shape[0], back_est.shape[1]), 0, dtype = float)
alphas = np.full(back_est.shape, np.nan, dtype = float)
#alphas_sum = np.full((back_est.shape[0], back_est.shape[1]), np.nan, dtype = float)

print(sup_arr.shape)

for a in range(num_configs):
  if a == int(Config - 1):
   for m in tqdm(range(len(m_range))):
    #if m != 189: continue
    #if m != 0: continue
    for d in range(len(d_range)):
      b_est = back_est[d, m, a]
      if b_est > 1e2: continue
      if np.isinf(np.log(b_est)): continue
      sup_arr[d, m, a], alphas[d, m, a] = GetS_up(b_est, 0.9, n=100)
      #if a == 0:
      #  b_est_sum = back_est_sum[d, m]
      #  if b_est_sum > 1e2: continue
      #  if np.isinf(np.log(b_est_sum)): continue
      #  sup_arr_sum[d, m], alphas_sum[d, m] = GetS_up(b_est_sum, 0.9, n=100)

bad_idx = np.logical_or(np.isinf(sup_arr), np.isnan(sup_arr))
minus_idx = sup_arr < 0.02
sup_arr[bad_idx] = 0
sup_arr[minus_idx] = 0

#bad_idx = np.logical_or(np.isinf(sup_arr_sum), np.isnan(sup_arr_sum))
#minus_idx = sup_arr_sum < 0.02
#sup_arr_sum[bad_idx] = 0
#sup_arr_sum[minus_idx] = 0

s_ratio = sim_s_pass_hist / sup_arr
bad_idx = np.logical_or(np.isinf(s_ratio), np.isnan(s_ratio))
s_ratio[bad_idx] = 0
print(s_ratio.shape)

#s_ratio_sum = sim_s_pass_hist_sum / sup_arr_sum
#bad_idx = np.logical_or(np.isinf(s_ratio_sum), np.isnan(s_ratio_sum))
#s_ratio_sum[bad_idx] = 0
#print(s_ratio_sum.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = path+f'grid_scan_result_A{Station}_{Pol}_R{Config}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('m_range', data=m_range, compression="gzip", compression_opts=9)
hf.create_dataset('d_range', data=d_range, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.create_dataset('d_diff_range', data=d_diff_range, compression="gzip", compression_opts=9)
hf.create_dataset('dat_cut_diff_hist', data=dat_cut_diff_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_cut_diff_hist', data=sim_s_cut_diff_hist, compression="gzip", compression_opts=9)
hf.create_dataset('fit_line', data=fit_line, compression="gzip", compression_opts=9)
hf.create_dataset('fit_line_fill', data=fit_line_fill, compression="gzip", compression_opts=9)
hf.create_dataset('fit_param', data=fit_param, compression="gzip", compression_opts=9)
hf.create_dataset('back_est', data=back_est, compression="gzip", compression_opts=9)
#hf.create_dataset('back_est_sum', data=back_est_sum, compression="gzip", compression_opts=9)
hf.create_dataset('sup_arr', data=sup_arr, compression="gzip", compression_opts=9)
#hf.create_dataset('sup_arr_sum', data=sup_arr_sum, compression="gzip", compression_opts=9)
hf.create_dataset('alphas', data=alphas, compression="gzip", compression_opts=9)
#hf.create_dataset('alphas_sum', data=alphas_sum, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist', data=sim_s_pass_hist, compression="gzip", compression_opts=9)
#hf.create_dataset('sim_s_pass_hist_sum', data=sim_s_pass_hist_sum, compression="gzip", compression_opts=9)
hf.create_dataset('s_ratio', data=s_ratio, compression="gzip", compression_opts=9)
#hf.create_dataset('s_ratio_sum', data=s_ratio_sum, compression="gzip", compression_opts=9)
hf.close()

print(file_name)
print('done!')






