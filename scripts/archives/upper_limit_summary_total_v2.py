import numpy as np
import os, sys
import h5py
from tqdm import tqdm
import ROOT

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

Station = int(sys.argv[1])
num_pols = 2

d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'

hf_v = h5py.File(d_path+f'back_est_A{Station}_VPol_total_v3_no_scale.h5', 'r')
hf_h = h5py.File(d_path+f'back_est_A{Station}_HPol_total_v3_no_scale.h5', 'r')

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

hf_vv = h5py.File(d_path+f'proj_scan_A{Station}_VPol_total_v3.h5', 'r')
hf_hh = h5py.File(d_path+f'proj_scan_A{Station}_HPol_total_v3.h5', 'r')
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

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Upper_Limit_A{Station}_total_v3_no_scale.h5'
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
hf.close()
print('done! file is in:',path+file_name, size_checker(path+file_name))

