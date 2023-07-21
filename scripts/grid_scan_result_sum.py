import os, sys
import numpy as np
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
import math

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import get_path_info_v2
from tools.ara_run_manager import file_sorter

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
if Station == 2:
    num_configs = 7
if Station == 3:
    num_configs = 9

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/grid_scan_result_A{Station}_*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

hf = h5py.File(os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/grid_scan_A{Station}.h5', 'r')
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
sim_s_pass_hist = hf['sim_s_pass_hist'][:]
print(sim_s_pass_hist.shape)
sim_s_pass_hist_sum  = np.nansum(sim_s_pass_hist, axis = -1)
d_diff_range = d_range[1:]

dat_cut_diff_hist = np.full((len(d_range) - 1, len(m_range), 2, num_configs), 0, dtype = float)
sim_s_cut_diff_hist = np.full((len(d_range) - 1, len(m_range), 2, num_configs), 0, dtype = float)
fit_line = np.full(dat_cut_diff_hist.shape, np.nan, dtype = float)
fit_line_fill = np.copy(fit_line)
fit_param = np.full((2, len(m_range), 2, num_configs), np.nan, dtype = float)
back_est = np.full((len(d_range), len(m_range), 2, num_configs), np.nan, dtype = float)
sup_arr = np.full(back_est.shape, 0, dtype = float)
alphas = np.full(back_est.shape, np.nan, dtype = float)
s_ratio = np.copy(sup_arr)

pol_type = ['V', 'H']

for r in tqdm(range(len(d_run_tot))):

  #if r <10:

    try:
        hf = h5py.File(d_list[r], 'r')
    except OSError:
        print(d_list[r])
        continue
    con_idx = int(get_path_info_v2(d_list[r], '_R', '.h5')) - 1
    pol = str(get_path_info_v2(d_list[r], f'A{Station}_', '_R'))   
    pol_idx = pol_type.index(pol)
    print(con_idx, pol, pol_idx) 

    print(np.nansum(hf['sup_arr'][:]), np.any(np.isnan(hf['sup_arr'][:])), np.any(np.isinf(hf['sup_arr'][:])))
    print(np.nansum(hf['s_ratio'][:]), np.any(np.isnan(hf['s_ratio'][:])), np.any(np.isinf(hf['s_ratio'][:])))

    dat_cut_diff_hist[:, :, pol_idx, con_idx] = hf['dat_cut_diff_hist'][:, :, con_idx]
    sim_s_cut_diff_hist[:, :, pol_idx, con_idx] = hf['sim_s_cut_diff_hist'][:, :, con_idx]  
    fit_line[:, :, pol_idx, con_idx] = hf['fit_line'][:, :, con_idx]
    fit_line_fill[:, :, pol_idx, con_idx] = hf['fit_line_fill'][:, :, con_idx]
    fit_param[:, :, pol_idx, con_idx] = hf['fit_param'][:, :, con_idx]
    back_est[:, :, pol_idx, con_idx] = hf['back_est'][:, :, con_idx]
    sup_arr[:, :, pol_idx, con_idx] = hf['sup_arr'][:, :, con_idx]
    alphas[:, :, pol_idx, con_idx] = hf['alphas'][:, :, con_idx]
    s_ratio[:, :, pol_idx, con_idx] = hf['s_ratio'][:, :, con_idx]

back_est_sum = np.nansum(back_est, axis = -1)
sup_arr_sum = np.full((back_est.shape[0], back_est.shape[1], back_est.shape[2]), 0, dtype = float)
alphas_sum = np.full((back_est.shape[0], back_est.shape[1], back_est.shape[2]), np.nan, dtype = float)
for p in range(2):
  for m in tqdm(range(len(m_range))):
    for d in range(len(d_range)):
        b_est_sum = back_est_sum[d, m, p]
        if b_est_sum > 1e2: continue
        if np.isinf(np.log(b_est_sum)): continue
        sup_arr_sum[d, m, p], alphas_sum[d, m, p] = GetS_up(b_est_sum, 0.9, n=100)

bad_idx = np.logical_or(np.isinf(sup_arr_sum), np.isnan(sup_arr_sum))
minus_idx = sup_arr_sum < 0.02
sup_arr_sum[bad_idx] = 0
sup_arr_sum[minus_idx] = 0

s_ratio_sum = sim_s_pass_hist_sum / sup_arr_sum
bad_idx = np.logical_or(np.isinf(s_ratio_sum), np.isnan(s_ratio_sum))
s_ratio_sum[bad_idx] = 0
print(s_ratio_sum.shape)

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = path+f'grid_scan_result_A{Station}.h5'
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
hf.create_dataset('back_est_sum', data=back_est_sum, compression="gzip", compression_opts=9)
hf.create_dataset('sup_arr', data=sup_arr, compression="gzip", compression_opts=9)
hf.create_dataset('sup_arr_sum', data=sup_arr_sum, compression="gzip", compression_opts=9)
hf.create_dataset('alphas', data=alphas, compression="gzip", compression_opts=9)
hf.create_dataset('alphas_sum', data=alphas_sum, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist', data=sim_s_pass_hist, compression="gzip", compression_opts=9)
hf.create_dataset('sim_s_pass_hist_sum', data=sim_s_pass_hist_sum, compression="gzip", compression_opts=9)
hf.create_dataset('s_ratio', data=s_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('s_ratio_sum', data=s_ratio_sum, compression="gzip", compression_opts=9)
hf.close()

print(file_name)
print('done!')






