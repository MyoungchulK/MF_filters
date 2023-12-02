import numpy as np
import os, sys
import h5py
from tqdm import tqdm
from scipy.interpolate import interp1d
import ROOT

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
scale = bool(int(sys.argv[2]))
if scale:
  scale_name = ''
else:
  scale_name = '_no_scale'


num_pols = 2

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

hf_v = h5py.File(d_path+f'back_est_A{Station}_VPol_total_v3{scale_name}.h5', 'r')
hf_h = h5py.File(d_path+f'back_est_A{Station}_HPol_total_v3{scale_name}.h5', 'r')
print(d_path+f'back_est_A{Station}_VPol_total_v3{scale_name}.h5')

slope_a_v = hf_v['slope_a'][:]
slope_a_h = hf_h['slope_a'][:]
slope_a = np.array([slope_a_v, slope_a_h])
print(slope_a_v.shape)
print(slope_a_h.shape)
print(slope_a.shape)
inercept_b_v = hf_v['inercept_b'][:]
inercept_b_h = hf_h['inercept_b'][:]
inercept_b = np.array([inercept_b_v, inercept_b_h])
print(inercept_b_v.shape)
print(inercept_b_h.shape)
print(inercept_b.shape)
inercept_b_bins_v = hf_v['inercept_b_bins'][:]
inercept_b_bins_h = hf_h['inercept_b_bins'][:]
inercept_b_bins = np.array([inercept_b_bins_v, inercept_b_bins_h])
print(inercept_b_bins_v.shape)
print(inercept_b_bins_h.shape)
print(inercept_b_bins.shape)
map_d_bins_v = hf_v[f'map_d_bins'][:]
map_d_bins_h = hf_h[f'map_d_bins'][:]
map_d_bins = np.array([map_d_bins_v, map_d_bins_h])
print(map_d_bins_v.shape)
print(map_d_bins_h.shape)
print(map_d_bins.shape)
map_d_bin_center_v = hf_v[f'map_d_bin_center'][:]
map_d_bin_center_h = hf_h[f'map_d_bin_center'][:]
map_d_bin_center = np.array([map_d_bin_center_v, map_d_bin_center_h])
print(map_d_bin_center_v.shape)
print(map_d_bin_center_h.shape)
print(map_d_bin_center.shape)

num_slos = len(slope_a_v)
map_d_len = len(inercept_b_v)

back_median_v = hf_v['back_median'][:]
back_median_h = hf_h['back_median'][:]
back_medi = np.array([back_median_v, back_median_h])
back_err_v = hf_v['back_err'][:]
back_err_h = hf_h['back_err'][:]
back_err = np.array([back_err_v, back_err_h])
print(back_median_v.shape)
print(back_median_h.shape)
print(back_medi.shape)
print(back_err_v.shape)
print(back_err_h.shape)
print(back_err.shape)

back_median_v_n = hf_v['back_median_n'][:]
back_median_h_n = hf_h['back_median_n'][:]
back_medi_n = np.array([back_median_v_n, back_median_h_n])
back_err_v_n = hf_v['back_err_n'][:]
back_err_h_n = hf_h['back_err_n'][:]
back_err_n = np.array([back_err_v_n, back_err_h_n])
print(back_median_v_n.shape)
print(back_median_h_n.shape)
print(back_medi_n.shape)
print(back_err_v_n.shape)
print(back_err_h_n.shape)
print(back_err_n.shape)

p_val_net_v = hf_v['p_val_net'][:]
p_val_net_h = hf_h['p_val_net'][:]
p_val = np.array([p_val_net_v, p_val_net_h])
print(p_val_net_v.shape)
print(p_val_net_h.shape)
print(p_val.shape)

hf_vv = h5py.File(d_path+f'proj_scan_A{Station}_VPol_total_v3_spice.h5', 'r')
hf_hh = h5py.File(d_path+f'proj_scan_A{Station}_HPol_total_v3_spice.h5', 'r')
print('d_path+f'proj_scan_A{Station}_VPol_total_v3_spice.h5'')
map_s_pass_mean_v = hf_vv['map_s_pass_mean'][:]
map_s_cut_mean_v = hf_vv['map_s_cut_mean'][:]
map_s_pass_mean_h = hf_hh['map_s_pass_mean'][:]
map_s_cut_mean_h = hf_hh['map_s_cut_mean'][:]
map_s_pass_mean = np.array([map_s_pass_mean_v, map_s_pass_mean_h])
map_s_cut_mean = np.array([map_s_cut_mean_v, map_s_cut_mean_h])
print(map_s_pass_mean_v.shape)
print(map_s_cut_mean_v.shape)
print(map_s_pass_mean_h.shape)
print(map_s_cut_mean_h.shape)
print(map_s_pass_mean.shape)
print(map_s_cut_mean.shape)

num_trials = 10001
bin_width = 100
bin_edges_o = np.full((2, num_pols, map_d_len, num_slos), np.nan, dtype = float)
bin_edges_u = np.copy(bin_edges_o)
nobs_hist = np.full((bin_width, num_pols, map_d_len, num_slos), np.nan, dtype = int)
upl_hist = np.copy(nobs_hist)
upl_mean = np.full((num_pols, map_d_len, num_slos), np.nan, dtype = float)
upl_tot_bins = np.linspace(0, 50, 25 + 1)
upl_tot_bin_center = (upl_tot_bins[1:] + upl_tot_bins[:-1]) / 2
upl_tot_len = len(upl_tot_bin_center)
upl_tot_hist = np.full((upl_tot_len, num_pols, map_d_len, num_slos), 0, dtype = int)

fc = ROOT.TFeldmanCousins(0.90) 

for s in tqdm(range(num_slos)):
        for d in range(map_d_len):
            for p in range(num_pols):

                if back_medi[p, d, s] > 10: continue

                nobs = np.random.poisson(back_medi[p, d, s], num_trials)
                nobs_uni, nobs_counts = np.unique(nobs, return_counts = True)  
                nobs_len = len(nobs_counts)
                upl_uni = np.full((nobs_len), np.nan, dtype = float)
                del nobs

                for n in range(nobs_len):
                    upl_uni[n] = fc.CalculateUpperLimit(nobs_uni[n], back_medi[p, d, s])
                upl_mean[p, d, s] = np.average(upl_uni, weights = nobs_counts)                

                nobs_min = np.nanmin(nobs_uni)
                nobs_max = np.nanmax(nobs_uni)
                if nobs_len == 1:
                    nobs_min -= 1 
                    nobs_max += 1 
                bins_o = np.linspace(nobs_min, nobs_max, bin_width + 1)
                bin_edges_o[:, p, d, s] = np.array([nobs_min, nobs_max])
                nobs_hist[:, p, d, s] = np.histogram(nobs_uni, weights = nobs_counts, bins = bins_o)[0].astype(int)
                del nobs_uni, nobs_min, nobs_max, bins_o

                upl_min = np.nanmin(upl_uni)
                upl_max = np.nanmax(upl_uni)
                if nobs_len == 1:
                    upl_min -= 1
                    upl_max += 1
                bins_u = np.linspace(upl_min, upl_max, bin_width + 1)
                bin_edges_u[:, p, d, s] = np.array([upl_min, upl_max])
                upl_hist[:, p, d, s] = np.histogram(upl_uni, weights = nobs_counts, bins = bins_u)[0].astype(int)                
                upl_tot_hist[:, p, d, s] = np.histogram(upl_uni, weights = nobs_counts, bins = upl_tot_bins)[0].astype(int)                
                del nobs_counts, nobs_len, upl_uni, upl_min, upl_max, bins_u 

s_s_up = map_s_pass_mean / upl_mean

p_val_cut0 = np.repeat(p_val[:, np.newaxis, :],  map_d_len, axis = 1)
p_val_cut = p_val_cut0 <= 0.05

s_s_up_cut = np.copy(s_s_up)
s_s_up_cut[p_val_cut] = np.nan

bins_s = np.linspace(0 - 0.5, 179 + 0.5, num_slos + 1)
bin_center_s = (bins_s[1:] + bins_s[:-1]) / 2
m_bins = np.linspace(0, 100, map_d_len + 1)
m_bin_center = (m_bins[1:] + m_bins[:-1]) / 2
s_s_up_map = np.full((num_pols, map_d_len, num_slos), np.nan, dtype = float)
s_s_up_cut_map = np.copy(s_s_up_map)
for p in range(num_pols):
  for s in tqdm(range(num_slos)):
    if s == 0: continue
    f = interp1d(inercept_b[p, :, s], s_s_up[p, :, s], fill_value="extrapolate")
    s_s_up_map[p, :, s] = f(m_bin_center)
    ff = interp1d(inercept_b[p, :, s], s_s_up_cut[p, :, s], fill_value="extrapolate")
    s_s_up_cut_map[p, :, s] = ff(m_bin_center)

p_len = num_pols
s_len = num_slos
s_max_idx = np.full((p_len, 2), 0, dtype = int)
s_max_ang_idx = np.full((p_len, 2, s_len), 0, dtype = int)
s_max = np.full((p_len, 2), np.nan, dtype = float)
s_max_ang = np.full((p_len, 2, s_len), np.nan, dtype = float)
back_est_opt = np.full((p_len), np.nan, dtype = float)
back_est_opt_err = np.full((p_len, 2), np.nan, dtype = float)
back_est_opt_n = np.full((p_len), np.nan, dtype = float)
back_est_opt_err_n = np.full((p_len, 2), np.nan, dtype = float)
slope_opt = np.copy(back_est_opt)
intercept_opt = np.copy(back_est_opt)
for p in range(p_len):
    max_idxs = np.where(s_s_up_cut[p, :, :] == np.nanmax(s_s_up_cut[p, :, :]))
    s_max_idx[p] = np.array([max_idxs[0][0], max_idxs[1][0]])
    if max_idxs[1][0] == 0: s_max[p, 0] = map_d_bin_center[p, :, :][max_idxs[0][0], max_idxs[1][0]]
    else: s_max[p, 0] = inercept_b[p, :, :][max_idxs[0][0], max_idxs[1][0]]
    s_max[p, 1] = s_s_up_cut[p, :, :][max_idxs[0][0], max_idxs[1][0]]

    for s in range(s_len):
        max_idxs_s = np.where(s_s_up[p, :, s] == np.nanmax(s_s_up[p, :, s]))[0]
        if len(max_idxs_s) == 0: continue
        s_max_ang_idx[p, :, s] = np.array([max_idxs_s[0], s])
        
        if max_idxs_s[0] == 0: s_max_ang[p, 0, s] = map_d_bin_center[p, :, s][max_idxs_s[0]]
        else: s_max_ang[p, 0, s] = inercept_b[p, :, s][max_idxs_s[0]]
        s_max_ang[p, 1, s] = s_s_up[p, :, s][max_idxs_s[0]]

    x_idx = max_idxs[0][0]
    y_idx = max_idxs[1][0]
    back_est_opt[p] = back_medi[p, x_idx, y_idx]
    back_est_opt_err[p] = back_err[p, :, x_idx, y_idx]
    back_est_opt_n[p] = back_medi_n[p, x_idx, y_idx]
    back_est_opt_err_n[p] = back_err_n[p, :, x_idx, y_idx]
    slope_opt[p] = slope_a[p, y_idx]
    intercept_opt[p] = inercept_b[p, x_idx, y_idx]

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Upper_Limit_A{Station}_total_v3{scale_name}_spice.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('slope_a', data=slope_a, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b', data=inercept_b, compression="gzip", compression_opts=9)
hf.create_dataset('inercept_b_bins', data=inercept_b_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('back_medi', data=back_medi, compression="gzip", compression_opts=9)
hf.create_dataset('back_err', data=back_err, compression="gzip", compression_opts=9)
hf.create_dataset('back_medi_n', data=back_medi_n, compression="gzip", compression_opts=9)
hf.create_dataset('back_err_n', data=back_err_n, compression="gzip", compression_opts=9)
hf.create_dataset('p_val', data=p_val, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass_mean', data=map_s_pass_mean, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut_mean', data=map_s_cut_mean, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_o', data=bin_edges_o, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_u', data=bin_edges_u, compression="gzip", compression_opts=9)
hf.create_dataset('nobs_hist', data=nobs_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_hist', data=upl_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_mean', data=upl_mean, compression="gzip", compression_opts=9)
hf.create_dataset('upl_tot_bins', data=upl_tot_bins, compression="gzip", compression_opts=9)
hf.create_dataset('upl_tot_bin_center', data=upl_tot_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('upl_tot_hist', data=upl_tot_hist, compression="gzip", compression_opts=9)
hf.create_dataset('s_s_up', data=s_s_up, compression="gzip", compression_opts=9)
hf.create_dataset('p_val_cut', data=p_val_cut0, compression="gzip", compression_opts=9)
hf.create_dataset('s_s_up_cut', data=s_s_up_cut, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('m_bins', data=m_bins, compression="gzip", compression_opts=9)
hf.create_dataset('m_bin_center', data=m_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('s_s_up_map', data=s_s_up_map, compression="gzip", compression_opts=9)
hf.create_dataset('s_s_up_cut_map', data=s_s_up_cut_map, compression="gzip", compression_opts=9)
hf.create_dataset('s_max_idx', data=s_max_idx, compression="gzip", compression_opts=9)
hf.create_dataset('s_max_ang_idx', data=s_max_ang_idx, compression="gzip", compression_opts=9)
hf.create_dataset('s_max', data=s_max, compression="gzip", compression_opts=9)
hf.create_dataset('s_max_ang', data=s_max_ang, compression="gzip", compression_opts=9)
hf.create_dataset('back_est_opt', data=back_est_opt, compression="gzip", compression_opts=9)
hf.create_dataset('back_est_opt_err', data=back_est_opt_err, compression="gzip", compression_opts=9)
hf.create_dataset('back_est_opt_n', data=back_est_opt_n, compression="gzip", compression_opts=9)
hf.create_dataset('back_est_opt_err_n', data=back_est_opt_err_n, compression="gzip", compression_opts=9)
hf.create_dataset('slope_opt', data=slope_opt, compression="gzip", compression_opts=9)
hf.create_dataset('intercept_opt', data=intercept_opt, compression="gzip", compression_opts=9)
hf.close()
print('done! file is in:',path+file_name, size_checker(path+file_name))
