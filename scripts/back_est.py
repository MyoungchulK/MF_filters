import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from scipy.optimize import curve_fit

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
Config = int(sys.argv[2])
ppol = int(sys.argv[3])
if ppol == 0: Pol = 'VPol'
if ppol == 1: Pol = 'HPol'

dpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
rpath = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

wfname = f'Data_Summary_v16_A{Station}.h5'
hf_d = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Data_Summary_Qual_v10_A{Station}.h5'
hf_q = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Data_Summary_Live_v6_A{Station}.h5'
hf_l = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Sim_Summary_signal_v19_A{Station}.h5'
hf_s = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Sim_Summary_signal_Qual_v13_A{Station}.h5'
hf_s_q = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Sim_Summary_noise_v19_A{Station}.h5'
hf_n = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)
wfname = f'Sim_Summary_noise_Qual_v13_A{Station}.h5'
hf_n_q = h5py.File(dpath+wfname, 'r')
print(dpath+wfname)

## config
config_d = hf_d[f'configs'][:]
print(config_d.shape)
con_ep = hf_d['con_ep'][:]
print(con_ep.shape)
configs_b = hf_l['configs'][:]
print(configs_b.shape)
livetime_d = hf_l['livetime'][:]
print(livetime_d.shape)
num_evts_s = 100
config_s = hf_s_q['config'][:]
config_s = np.repeat(config_s[:, np.newaxis], num_evts_s, axis = 1)
print(config_s.shape)
flavor_s = hf_s_q['flavor'][:]
flavor_s = np.repeat(flavor_s[:, np.newaxis], num_evts_s, axis = 1)
print(flavor_s.shape)
sig_in_wide = hf_s_q['sig_in_wide'][:] == 0
print(sig_in_wide.shape)
ray_in_air = hf_s_q['ray_in_air'][:] == 0
print(ray_in_air.shape)
both_idx = np.logical_or(sig_in_wide, sig_in_wide)
evt_rate_s = hf_s_q['evt_rate'][:]
evt_rate_s[both_idx] = 0
print(evt_rate_s.shape)
del sig_in_wide, ray_in_air
num_evts_n = 1000
config_n = hf_n['config'][:]
config_n = np.repeat(config_n[:, np.newaxis], num_evts_n, axis = 1)
print(config_n.shape)

if Station == 2: num_configs = 7
if Station == 3: num_configs = 9
pol_name = ['VPol', 'HPol', 'VHPol']
pol_idx = pol_name.index(Pol)
trig_name = ['RF', 'Cal', 'Soft']
trig = 'RF'
trig_idx = trig_name.index(trig)
fla_name = ['NuE', 'NuMu', 'NuTau']
num_flas = len(fla_name)

qual_s = hf_s_q['qual'][:] != 0
print(qual_s.shape)
qual_cw_s = hf_s_q['qual_cw'][:] != 0
print(qual_cw_s.shape)
qual_op_s = hf_s_q['qual_op'][:] != 0
print(qual_op_s.shape)
qual_cp_s = hf_s_q['qual_cp'][:] != 0
print(qual_cp_s.shape)
qual_corr_t_s = hf_s_q['qual_corr_t'][:] != 0
print(qual_corr_t_s.shape)
qual_corr_z_s = hf_s_q['qual_corr_z'][:] != 0
print(qual_corr_z_s.shape)
qual_ver_t_s = hf_s_q['qual_ver_t'][:] != 0
print(qual_ver_t_s.shape)
qual_ver_z_s = hf_s_q['qual_ver_z'][:] != 0
print(qual_ver_z_s.shape)
quals_s = np.any((qual_s, qual_cw_s, qual_op_s, qual_cp_s, qual_corr_t_s, qual_corr_z_s, qual_ver_t_s, qual_ver_z_s), axis = 0)
print(qual_s.shape)
del qual_s, qual_cw_s, qual_op_s, qual_cp_s, qual_corr_t_s, qual_corr_z_s, qual_ver_t_s, qual_ver_z_s

qual_n = hf_n_q['qual'][:] != 0
print(qual_n.shape)
qual_cw_n = hf_n_q['qual_cw'][:] != 0
print(qual_cw_n.shape)
qual_op_n = hf_n_q['qual_op'][:] != 0
print(qual_op_n.shape)
qual_cp_n = hf_n_q['qual_cp'][:] != 0
print(qual_cp_n.shape)
qual_corr_t_n = hf_n_q['qual_corr_t'][:] != 0
print(qual_corr_t_n.shape)
qual_corr_z_n = hf_n_q['qual_corr_z'][:] != 0
print(qual_corr_z_n.shape)
qual_ver_t_n = hf_n_q['qual_ver_t'][:] != 0
print(qual_ver_t_n.shape)
qual_ver_z_n = hf_n_q['qual_ver_z'][:] != 0
print(qual_ver_z_n.shape)
quals_n = np.any((qual_n, qual_cw_n, qual_op_n, qual_cp_n, qual_corr_t_n, qual_corr_z_n, qual_ver_t_n, qual_ver_z_n), axis = 0)
print(quals_n.shape)
del qual_n, qual_cw_n, qual_op_n, qual_cp_n, qual_corr_t_n, qual_corr_z_n, qual_ver_t_n, qual_ver_z_n

trig_d = hf_d['trig_ep'][:] != trig_idx
print(trig_d.shape)
qual_ep_no = np.full((trig_d.shape), False, dtype = bool)
qual_ep = np.logical_or(trig_d, hf_q['qual_ep'][:] != 0)
print(qual_ep.shape)
qual_ep_cw = np.logical_or(trig_d, hf_q['qual_ep_cw'][:] != 0)
print(qual_ep_cw.shape)
qual_ep_op = np.logical_or(trig_d, hf_q['qual_ep_op'][:] != 0)
print(qual_ep_op.shape)
qual_ep_cp = np.logical_or(trig_d, hf_q['qual_ep_cp'][:] != 0)
print(qual_ep_cp.shape)
qual_ep_corr_t = np.logical_or(trig_d, hf_q['qual_ep_corr_t'][:] != 0)
print(qual_ep_corr_t.shape)
qual_ep_corr_z = np.logical_or(trig_d, hf_q['qual_ep_corr_z'][:] != 0)
print(qual_ep_corr_z.shape)
qual_ep_ver_t = np.logical_or(trig_d, hf_q['qual_ep_ver_t'][:] != 0)
print(qual_ep_ver_t.shape)
qual_ep_ver_z = np.logical_or(trig_d, hf_q['qual_ep_ver_z'][:] != 0)
print(qual_ep_ver_z.shape)
quals_d = np.any((qual_ep_no, qual_ep, qual_ep_cw, qual_ep_op, qual_ep_cp, qual_ep_corr_t, qual_ep_corr_z, qual_ep_ver_t, qual_ep_ver_z), axis = 0)
print(quals_d.shape)
del qual_ep_no, qual_ep, qual_ep_cw, qual_ep_op, qual_ep_cp, qual_ep_corr_t, qual_ep_corr_z, qual_ep_ver_t, qual_ep_ver_z

mf_val_s = hf_s['mf_max'][:, pol_idx]
mf_val_s[quals_s] = np.nan
print(mf_val_s.shape)
corr_val_s = hf_s['coef_max'][:, pol_idx]
corr_val_s[quals_s] = np.nan
print(corr_val_s.shape)
evt_rate_s[quals_s] = np.nan
print(evt_rate_s.shape)

mf_val_n = hf_n['mf_max'][:, pol_idx]
mf_val_n[quals_n] = np.nan
print(mf_val_n.shape)
corr_val_n = hf_n['coef_max'][:, pol_idx]
corr_val_n[quals_n] = np.nan
print(corr_val_n.shape)

mf_val_d = hf_d['mf_max'][pol_idx]
mf_val_d[quals_d] = np.nan
print(mf_val_d.shape)
corr_val_d = hf_d['coef_max'][pol_idx]
corr_val_d[quals_d] = np.nan
print(corr_val_d.shape)

def log_linear_fit(x, a, b):
  xmin = x[0]
  y = a *np.exp(-b * (x - xmin))

  return y

use_debug = False

s_ang = np.arange(91, dtype =int).astype(float)
slope = np.tan(np.radians(90) + (np.pi/180) * np.tan(np.radians(np.arange(0,-91,-1))))
slope_ang = np.degrees(np.arctan(slope))[1:-1]
s_ang = np.concatenate((s_ang, slope_ang))
s_ang = np.sort(s_ang)
s_width = np.abs(s_ang[1] - s_ang[0]) / 2
s_rad = np.radians(s_ang)
slope_m = np.tan(np.radians(s_ang + 90))
bins_s = np.linspace(0 - 0.5, 179 + 0.5, len(s_ang) + 1)
bin_center_s = (bins_s[1:] + bins_s[:-1]) / 2
s_len = len(bin_center_s)

d_len = 400
bins_d_tot = np.linspace(0,100,400+1)
bin_center_d_tot = (bins_d_tot[1:] + bins_d_tot[:-1]) / 2
bin_idx = np.arange(d_len, dtype = int)

map_d = np.full((d_len, s_len, num_configs), 0, dtype = float)
map_d_fit = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_xmin = np.full((s_len, num_configs), np.nan, dtype = float)
map_d_xmin_swap = np.copy(map_d_xmin)
map_d_bins = np.full((d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_d_bins_swap = np.full((2, d_len + 1, s_len, num_configs), np.nan, dtype = float)
map_d_bins_swap[0] = slope_m[np.newaxis, :, np.newaxis]
map_d_bin_center_swap = np.full((2, d_len, s_len, num_configs), np.nan, dtype = float)
map_d_bin_center_swap[0] = slope_m[np.newaxis, :, np.newaxis]
map_d_pdf = np.copy(map_d_bin_center)
map_d_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_d_err = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov
map_s = np.full((d_len, s_len, num_flas, num_configs), np.nan, dtype = float)
map_s_pass_int = np.copy(map_s)
map_s_cut_int = np.copy(map_s)
map_n = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_n_fit = np.full((d_len, s_len, num_configs), np.nan, dtype = float)
map_n_xmin = np.full((s_len, num_configs), np.nan, dtype = float)
map_n_xmin_swap = np.copy(map_n_xmin)
map_n_pdf = np.copy(map_d_bin_center)
map_n_param = np.full((2, s_len, num_configs), np.nan, dtype = float)
map_n_err = np.full((2, 2, s_len, num_configs), np.nan, dtype = float) # cov

map_d_tot = np.copy(map_d)
map_d_pdf_tot = np.copy(map_d_bin_center)
map_n_pdf_tot = np.copy(map_d_bin_center)
map_s_tot = np.copy(map_s)
map_n_tot = np.copy(map_n)

norm_fac = np.full((num_configs), np.nan, dtype = float)
norm_fac_n = np.full((num_configs), np.nan, dtype = float)
livesec = np.full((num_configs), np.nan, dtype = float)
for c in tqdm(range(num_configs)):
  livesec[c] = np.nansum(livetime_d[:, 1][configs_b == int(c + 1)])
evt_tot = np.full((3, num_configs), np.nan, dtype = float)
print(map_d.shape)
print(map_s.shape)
print(livesec.shape)
print(evt_tot.shape)

#for c in tqdm(range(num_configs)):
for c in range(num_configs):
  print(c)
  if c != int(Config - 1): continue
  #if c != 1 and use_debug == True: continue
  con_idx = con_ep == int(c + 1)
  norm_fac[c] = np.count_nonzero(con_idx) # norm factor
  corr_val_d_c = corr_val_d[con_idx]
  mf_val_d_c = mf_val_d[con_idx]
  del con_idx

  dat_rad = np.arctan(mf_val_d_c / corr_val_d_c) # data angle
  dat_dis = np.sqrt(mf_val_d_c ** 2 + corr_val_d_c ** 2) # data distance
  del corr_val_d_c, mf_val_d_c

  for s in tqdm(range(len(s_ang))):
    if s != 1 and use_debug == True: continue
    dat_rad_proj = np.abs(s_rad[s] - dat_rad) # data and plane angle diff
    dat_dis_proj = dat_dis * np.cos(dat_rad_proj) # data projection into plane
    del dat_rad_proj

    # dynamic bins
    bin_min = np.nanmin(dat_dis_proj)
    bin_max = np.nanmax(dat_dis_proj)
    bin_max_1p5 = bin_max + (bin_max - bin_min) * 1.5
    map_d_bins[:, s, c] = np.linspace(bin_min, bin_max_1p5,400+1)
    map_d_bin_center[:, s, c] = (map_d_bins[1:, s, c] + map_d_bins[:-1, s, c]) / 2
    map_d_bins_swap[1, :, s, c] = map_d_bins[:, s, c] / np.cos(np.radians(90 - s_ang[s]))
    map_d_bin_center_swap[1, :, s, c] = map_d_bin_center[:, s, c] / np.cos(np.radians(90 - s_ang[s]))
    map_d_tot[:, s, c] = np.histogram(dat_dis_proj / np.cos(np.radians(90 - s_ang[s])), bins = bins_d_tot)[0] / norm_fac[c]
    del bin_min, bin_max, bin_max_1p5

    # histogram
    map_d_hist = np.histogram(dat_dis_proj, bins = map_d_bins[:, s, c])[0].astype(int)
    max_idx = np.nanargmax(map_d_hist)
    dat_f_idx = bin_idx[np.logical_and(map_d_hist > 0, bin_idx > max_idx)][-1]
    dat_f_idx_in = bin_idx[np.logical_and(map_d_hist < 1, bin_idx > max_idx)][0]
    dat_len = int(dat_f_idx_in - max_idx)
    dat_i_idx = int(max_idx + int(dat_len * 0.6))
    del max_idx, dat_len, dat_dis_proj, dat_f_idx_in

    # save data
    map_d_hist = map_d_hist.astype(float) / norm_fac[c]
    map_d[:, s, c] = map_d_hist
    map_d_fit[dat_i_idx:dat_f_idx, s, c] = map_d_hist[dat_i_idx:dat_f_idx]
    map_d_xmin[s, c] = map_d_bin_center[dat_i_idx:dat_f_idx, s, c][0]
    map_d_xmin_swap[s, c] = map_d_bin_center[dat_i_idx:dat_f_idx, s, c][0] / np.cos(np.radians(90 - s_ang[s]))

    # fitting
    dat_x = map_d_bin_center[dat_i_idx:dat_f_idx, s, c]
    dat_y = map_d_hist[dat_i_idx:dat_f_idx]
    try:
      popt, pcov = curve_fit(log_linear_fit, dat_x, dat_y)
    except RuntimeError:
      plt.yscale('log')
      plt.plot(map_d_bin_center[:, s, c], map_d[:, s, c])
      #plt.plot(map_d_bin_center[:, s, c], map_d_pdf[:, s, c])
      plt.axvline(map_d_bin_center[dat_i_idx, s, c], color = 'red')
      plt.axvline(map_d_bin_center[dat_f_idx, s, c], color = 'orangered')
      continue
    fit_x = map_d_bin_center[dat_i_idx:, s, c]
    map_d_pdf[dat_i_idx:, s, c] = log_linear_fit(fit_x, *popt)
    #tot_idx = np.logical_and(bin_center_d_tot >= map_d_bin_center[dat_i_idx, s, c], bin_center_d_tot <= map_d_bin_center[dat_f_idx, s, c])
    tot_idx = bin_center_d_tot >= (map_d_bin_center[dat_i_idx, s, c] / np.cos(np.radians(90 - s_ang[s])))
    if np.sum(tot_idx) != 0:
      map_d_pdf_tot[tot_idx, s, c] = log_linear_fit(bin_center_d_tot[tot_idx], *popt)
    map_d_param[:, s, c] = popt
    map_d_err[:, :, s, c] = pcov
    del map_d_hist, dat_f_idx, dat_i_idx, dat_x, dat_y, fit_x, popt, pcov
  del dat_rad, dat_dis

  con_n_idx = config_n == int(c + 1)
  norm_fac_n[c] = np.count_nonzero(con_n_idx) # norm factor
  corr_val_n_c = corr_val_n[con_n_idx]
  mf_val_n_c = mf_val_n[con_n_idx]
  del con_n_idx

  sim_rad_n = np.arctan(mf_val_n_c / corr_val_n_c) # data angle
  sim_dis_n = np.sqrt(mf_val_n_c ** 2 + corr_val_n_c ** 2) # data distance
  del corr_val_n_c, mf_val_n_c

  for s in tqdm(range(len(s_ang))):
    if s != 1 and use_debug == True: continue
    sim_rad_proj_n = np.abs(s_rad[s] - sim_rad_n) # data and plane angle diff
    sim_dis_proj_n = sim_dis_n * np.cos(sim_rad_proj_n) # data projection into plane
    del sim_rad_proj_n

    map_n_hist = np.histogram(sim_dis_proj_n, bins = map_d_bins[:, s, c])[0].astype(int)
    max_idx = np.nanargmax(map_n_hist)
    dat_f_idx = bin_idx[np.logical_and(map_n_hist > 0, bin_idx > max_idx)][-1]
    dat_f_idx_in = bin_idx[np.logical_and(map_n_hist < 1, bin_idx > max_idx)][0]
    dat_len = int(dat_f_idx_in - max_idx)
    dat_i_idx = int(max_idx + int(dat_len * 0.6))
    del max_idx, dat_len, dat_f_idx_in

    # save data
    map_n_hist = map_n_hist.astype(float) / norm_fac[c]
    map_n[:, s, c] = map_n_hist
    map_n_fit[dat_i_idx:dat_f_idx, s, c] = map_n_hist[dat_i_idx:dat_f_idx]
    map_n_xmin[s, c] = map_d_bin_center[dat_i_idx:dat_f_idx, s, c][0]
    map_n_xmin_swap[s, c] = map_d_bin_center[dat_i_idx:dat_f_idx, s, c][0] / np.cos(np.radians(90 - s_ang[s]))
    map_n_tot[:, s, c] = np.histogram(sim_dis_proj_n / np.cos(np.radians(90 - s_ang[s])), bins = bins_d_tot)[0] / norm_fac[c]

    # fitting
    dat_x = map_d_bin_center[dat_i_idx:dat_f_idx, s, c]
    dat_y = map_n_hist[dat_i_idx:dat_f_idx]
    try:
      popt, pcov = curve_fit(log_linear_fit, dat_x, dat_y)
    except RuntimeError:
      plt.yscale('log')
      plt.plot(map_d_bin_center[:, s, c], map_n[:, s, c])
      #plt.plot(map_d_bin_center[:, s, c], map_n_pdf[:, s, c])
      plt.axvline(map_d_bin_center[dat_i_idx, s, c], color = 'red')
      plt.axvline(map_d_bin_center[dat_f_idx, s, c], color = 'orangered')
      continue
    fit_x = map_d_bin_center[dat_i_idx:, s, c]
    map_n_pdf[dat_i_idx:, s, c] = log_linear_fit(fit_x, *popt)
    tot_idx = bin_center_d_tot >= (map_d_bin_center[dat_i_idx, s, c] / np.cos(np.radians(90 - s_ang[s])))
    if np.sum(tot_idx) != 0:
      map_n_pdf_tot[tot_idx, s, c] = log_linear_fit(bin_center_d_tot[tot_idx], *popt)
    map_n_param[:, s, c] = popt
    map_n_err[:, :, s, c] = pcov
    del map_n_hist, dat_f_idx, dat_i_idx, dat_x, dat_y, fit_x, popt, pcov
    del sim_dis_proj_n
  del sim_rad_n, sim_dis_n

  con_s_idx = config_s == int(c + 1)
  for f in range(3):
    idxs = np.logical_and(flavor_s == int(f + 1), con_s_idx)
    evt_cnt_s = evt_rate_s[idxs].flatten() * livesec[c]
    evt_tot[f, c] = np.nansum(evt_cnt_s)
    corr_val_s_c = corr_val_s[idxs]
    mf_val_s_c = mf_val_s[idxs]
    del idxs

    sim_rad = np.arctan(mf_val_s_c / corr_val_s_c) # data angle
    sim_dis = np.sqrt(mf_val_s_c ** 2 + corr_val_s_c ** 2) # data distance
    del corr_val_s_c, mf_val_s_c

    for s in tqdm(range(len(s_ang))):
      if s != 1 and use_debug == True: continue
      sim_rad_proj = np.abs(s_rad[s] - sim_rad) # data and plane angle diff
      sim_dis_proj = sim_dis * np.cos(sim_rad_proj) # data projection into plane
      del sim_rad_proj

      map_s[:, s, f, c] = np.histogram(sim_dis_proj, weights = evt_cnt_s, bins = map_d_bins[:, s, c])[0] / norm_fac[c]
      map_s_tot[:, s, f, c] = np.histogram(sim_dis_proj / np.cos(np.radians(90 - s_ang[s])), weights = evt_cnt_s, bins = bins_d_tot)[0] / norm_fac[c]
      for d in range(d_len):
        cut_idx = sim_dis_proj < map_d_bin_center[d, s, c]
        map_s_cut_int[d, s, f, c] = np.nansum(cut_idx.astype(float) * evt_cnt_s)
        map_s_pass_int[d, s, f, c] = np.nansum((~cut_idx).astype(float) * evt_cnt_s)
      del sim_dis_proj
    del sim_rad, sim_dis, evt_cnt_s
  del con_s_idx

live_days = livesec / (60 * 60 * 24)
evt_tot_mean = np.nanmean(evt_tot, axis = 0)
map_s_mean = np.nanmean(map_s, axis = 2)
map_s_tot_mean = np.nanmean(map_s_tot, axis = 2)
map_s_pass_int_mean = np.nanmean(map_s_pass_int, axis = 2)
map_s_cut_int_mean = np.nanmean(map_s_cut_int, axis = 2)
print(live_days)
print(evt_tot_mean)
print(map_s_mean.shape)

file_name = dpath+f'proj_scan_A{Station}_{Pol}_R{Config}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('s_rad', data=s_rad, compression="gzip", compression_opts=9)
hf.create_dataset('slope_m', data=slope_m, compression="gzip", compression_opts=9)
hf.create_dataset('bins_d_tot', data=bins_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_d_tot', data=bin_center_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d', data=map_d, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_fit', data=map_d_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_tot', data=map_d_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin', data=map_d_xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_xmin_swap', data=map_d_xmin_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins_swap', data=map_d_bins_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center_swap', data=map_d_bin_center_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf', data=map_d_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_pdf_tot', data=map_d_pdf_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_param', data=map_d_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_err', data=map_d_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_s', data=map_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot', data=map_s_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass_int', data=map_s_pass_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut_int', data=map_s_cut_int, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_mean', data=map_s_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_tot_mean', data=map_s_tot_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass_int_mean', data=map_s_pass_int_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut_int_mean', data=map_s_cut_int_mean, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac', data=norm_fac, compression="gzip", compression_opts=9)
hf.create_dataset('livesec', data=livesec, compression="gzip", compression_opts=9)
hf.create_dataset('live_days', data=live_days, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot', data=evt_tot, compression="gzip", compression_opts=9)
hf.create_dataset('evt_tot_mean', data=evt_tot_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_n', data=map_n, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_fit', data=map_n_fit, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin', data=map_n_xmin, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_xmin_swap', data=map_n_xmin_swap, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf', data=map_n_pdf, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_param', data=map_n_param, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_err', data=map_n_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_pdf_tot', data=map_n_pdf_tot, compression="gzip", compression_opts=9)
hf.create_dataset('map_n_tot', data=map_n_tot, compression="gzip", compression_opts=9)
hf.create_dataset('norm_fac_n', data=norm_fac_n, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',file_name, size_checker(file_name))
print('done!', file_name)
