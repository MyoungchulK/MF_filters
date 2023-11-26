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

hf_v = h5py.File(d_path+f'back_est_A{Station}_VPol_total_v2.h5', 'r')
hf_h = h5py.File(d_path+f'back_est_A{Station}_HPol_total_v2.h5', 'r')

s_ang = hf_v['s_ang'][:]
num_slos = len(s_ang)
bins_s = hf_v['bins_s'][:]
bin_center_s = hf_v['bin_center_s'][:]
print(s_ang.shape)
print(bins_s.shape)
print(bin_center_s.shape)

map_d_bins_v = hf_v['map_d_bins'][:]
map_d_bin_center_v = hf_v['map_d_bin_center'][:]
map_d_bins_h = hf_h['map_d_bins'][:]
map_d_bin_center_h = hf_h['map_d_bin_center'][:]
map_d_len = len(map_d_bin_center_h[:, 0])
map_d_bins = np.array([map_d_bins_v, map_d_bins_h])
map_d_bin_center = np.array([map_d_bin_center_v, map_d_bin_center_h])
print(map_d_bins_v.shape)
print(map_d_bin_center_v.shape)
print(map_d_bins_h.shape)
print(map_d_bin_center_h.shape)
print(map_d_bins.shape)
print(map_d_bin_center.shape)

slope_m = np.full((num_pols, map_d_len, len(bin_center_s)), np.nan, dtype = float)
slope_m[:] = np.tan(np.radians(s_ang + 90))[np.newaxis, np.newaxis, :]
intercept_d = map_d_bin_center / np.cos(np.radians(90 - s_ang))[np.newaxis, np.newaxis, :]
print(slope_m.shape)
print(intercept_d.shape)

back_median_v = hf_v['back_median'][:]
back_err_v = hf_v['back_err'][:]
back_median_h = hf_h['back_median'][:]
back_err_h = hf_h['back_err'][:]
back_medi = np.array([back_median_v, back_median_h])
back_err = np.array([back_err_v, back_err_h])
print(back_median_v.shape)
print(back_err_v.shape)
print(back_median_h.shape)
print(back_err_h.shape)
print(back_medi.shape)
print(back_err.shape)

hf_vv = h5py.File(d_path+f'proj_scan_A{Station}_VPol_total_v2.h5', 'r')
hf_hh = h5py.File(d_path+f'proj_scan_A{Station}_HPol_total_v2.h5', 'r')
map_s_pass_int_tot_v = hf_vv['map_s_pass_int_mean'][:]
map_s_cut_int_tot_v = hf_vv['map_s_cut_int_mean'][:]
map_s_pass_int_tot_h = hf_hh['map_s_pass_int_mean'][:]
map_s_cut_int_tot_h = hf_hh['map_s_cut_int_mean'][:]
map_s_pass = np.array([map_s_pass_int_tot_v, map_s_pass_int_tot_h])
map_s_cut = np.array([map_s_cut_int_tot_v, map_s_cut_int_tot_h])
print(map_s_pass_int_tot_v.shape)
print(map_s_cut_int_tot_v.shape)
print(map_s_pass_int_tot_h.shape)
print(map_s_cut_int_tot_h.shape)
print(map_s_pass.shape)
print(map_s_cut.shape)

num_trials = 10001
bin_width = 100
bin_edges_o = np.full((2, num_pols, map_d_len, num_slos), np.nan, dtype = float)
bin_edges_u = np.copy(bin_edges_o)
nobs_hist = np.full((bin_width, num_pols, map_d_len, num_slos), np.nan, dtype = int)
upl_hist = np.copy(nobs_hist)
upl_mean = np.full((num_pols, map_d_len, num_slos), np.nan, dtype = float)

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
                del nobs_counts, nobs_len, upl_uni, upl_min, upl_max, bins_u 

s_up_s = upl_mean / map_s_pass

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Upper_Limit_A{Station}_total_v2.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('s_ang', data=s_ang, compression="gzip", compression_opts=9)
hf.create_dataset('bins_s', data=bins_s, compression="gzip", compression_opts=9)
hf.create_dataset('bin_center_s', data=bin_center_s, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bins', data=map_d_bins, compression="gzip", compression_opts=9)
hf.create_dataset('map_d_bin_center', data=map_d_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('slope_m', data=slope_m, compression="gzip", compression_opts=9)
hf.create_dataset('intercept_d', data=intercept_d, compression="gzip", compression_opts=9)
hf.create_dataset('back_medi', data=back_medi, compression="gzip", compression_opts=9)
hf.create_dataset('back_err', data=back_err, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_pass', data=map_s_pass, compression="gzip", compression_opts=9)
hf.create_dataset('map_s_cut', data=map_s_cut, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_o', data=bin_edges_o, compression="gzip", compression_opts=9)
hf.create_dataset('bin_edges_u', data=bin_edges_u, compression="gzip", compression_opts=9)
hf.create_dataset('nobs_hist', data=nobs_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_hist', data=upl_hist, compression="gzip", compression_opts=9)
hf.create_dataset('upl_mean', data=upl_mean, compression="gzip", compression_opts=9)
hf.create_dataset('s_up_s', data=s_up_s, compression="gzip", compression_opts=9)
hf.close()
print('done! file is in:',path+file_name, size_checker(path+file_name))

