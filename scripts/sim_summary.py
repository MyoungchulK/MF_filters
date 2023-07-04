import numpy as np
import os, sys
from glob import glob
import h5py
from tqdm import tqdm

curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import file_sorter
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_example_run
from tools.ara_known_issue import known_issue_loader

Station = int(sys.argv[1])
Type = str(sys.argv[2])

known_issue = known_issue_loader(Station)

# sort
d_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/sub_info_sim/*{Type}*'
d_list, d_run_tot, d_run_range, d_len = file_sorter(d_path)
del d_run_range

s_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/snr_sim/'
r_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/reco_ele_sim/'
m_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/mf_sim/'
c_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/csw_sim/'
q_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/qual_cut_sim/'

if Type == 'signal':
    num_evts = 100
if Type == 'noise':
    num_evts = 1000
num_ants = 16
evt_num = np.arange(num_evts, dtype = int)
pol_num = np.arange(2, dtype = int)
theta_bin = np.array([60, 40, 20, 0, 20, 10, 0, -10, -20, -40, -60], dtype = int)
phi_bin = np.array([-150, -90, -30, 30, 90, 150], dtype = int)
ele_bin = 90 - np.linspace(0.5, 179.5, 179 + 1)
#sur_idx = ele_bin > 35
sur_idx = np.full((180), False, dtype = bool)
sur_idx[50:53] = True

pnu = np.full((d_len, num_evts), np.nan, dtype = float)
sim_run = np.full((d_len), 0, dtype = int)
config = np.copy(sim_run)
flavor = np.copy(sim_run)
exponent = np.full((d_len, 2), 0, dtype = int)
radius = np.full((d_len), np.nan, dtype = float)
inu_thrown = np.copy(radius)
weight = np.copy(pnu)
probability = np.copy(pnu)
nuflavorint = np.copy(pnu)
nu_nubar = np.copy(pnu)
currentint = np.copy(pnu)
elast_y = np.copy(pnu)
posnu = np.full((d_len, 6, num_evts), np.nan, dtype = float)
nnu = np.copy(posnu)
posnu_antcen = np.copy(posnu)
rec_ang = np.full((d_len, 2, num_ants, num_evts), np.nan, dtype = float)
view_ang = np.copy(rec_ang)
arrival_time = np.copy(rec_ang)
evt_rate = np.copy(pnu)
coef = np.full((d_len, 2, 2, 2, num_evts), np.nan, dtype = float)
coord = np.full((d_len, 2, 2, 2, 2, num_evts), np.nan, dtype = float) 
coef_max = np.full((d_len, 2, num_evts), np.nan, dtype = float) 
coef_ratio = np.full((d_len, 2, num_evts), np.nan, dtype = float)
coord_max = np.full((d_len, 2, 3, num_evts), np.nan, dtype = float)
mf_max = np.full((d_len, 2, num_evts), np.nan, dtype = float)
mf_ser_max = np.full((d_len, 2, 2, num_evts), np.nan, dtype = float)
mf_ratio = np.full((d_len, 2, num_evts), np.nan, dtype = float)
snr = np.full((d_len, num_ants, num_evts), np.nan, dtype = float)
snr_max = np.full((d_len, 2, num_evts), np.nan, dtype = float)
qual = np.full((d_len, num_evts, 4), 0, dtype = int)
qual_tot = np.full((d_len, num_evts), 0, dtype = int)
sig_in = np.full((d_len, num_evts), 0, dtype = int)
sig_in_wide = np.full((d_len, num_evts), 0, dtype = int)
signal_bin = np.copy(rec_ang)
hill_max_idx = np.full((d_len, 2, 2, num_evts), np.nan, dtype = float)
hill_max = np.copy(hill_max_idx)
snr_csw = np.copy(hill_max_idx)
cdf_avg = np.copy(hill_max_idx)
slope = np.copy(hill_max_idx)
intercept = np.copy(hill_max_idx)
r_value = np.copy(hill_max_idx)
p_value = np.copy(hill_max_idx)
std_err = np.copy(hill_max_idx)
ks = np.copy(hill_max_idx)
nan_flag = np.full((d_len, 2, 2, num_evts), 0, dtype = int)

z_bins = np.linspace(-90, 90, 180 + 1)
z_bin_center = (z_bins[1:] + z_bins[:-1]) / 2
a_bins = np.linspace(-180, 180, 360 + 1)
a_bin_center = (a_bins[1:] + a_bins[:-1]) / 2
c_bins = np.linspace(0, 1.2, 120 + 1)
c_bin_center = (c_bins[1:] + c_bins[:-1]) / 2
m_bins = np.linspace(0, 100, 200 + 1)
m_bin_center = (m_bins[1:] + m_bins[:-1]) / 2

rad_o = np.array([41, 300], dtype = float)

nfour = float(2048 / 2 / 2 * 0.5)

for r in tqdm(range(len(d_run_tot))):
    
  #if r > 3310:

    try:
        hf = h5py.File(d_list[r], 'r')
    except FileNotFoundError:
        print(d_list[r])
        continue
    cons = hf['config'][:]
    sim_run[r] = cons[1]
    config[r] = cons[2]
    flavor[r] = cons[4]
    radius[r] = hf['radius'][:]
    pnu[r] = hf['pnu'][:]
    inu_thrown[r] = hf['inu_thrown'][-1]
    weight[r] = hf['weight'][:]
    probability[r] = hf['probability'][:]
    nuflavorint[r] = hf['nuflavorint'][:]
    nu_nubar[r] = hf['nu_nubar'][:]
    currentint[r] = hf['currentint'][:]
    elast_y[r] = hf['elast_y'][:]
    posnu[r] = hf['posnu'][:]
    posnu_antcen[r] = hf['posnu_antcen'][:]
    nnu[r] = hf['nnu'][:]
    rec_ang[r] = hf['rec_ang'][:]
    view_ang[r] = hf['view_ang'][:]
    arrival_time[r] = hf['arrival_time'][:]
    exponent[r] = hf['exponent_range'][:]
    wf_time = hf['wf_time'][:]
    wf_dege = np.array([wf_time[0] - 0.5, wf_time[-1] + 0.5])
    wf_dege_wide = np.array([wf_time[0] - nfour - 0.5, wf_time[-1] + nfour + 0.5])
    sig_bin = hf['signal_bin'][:]
    sig_in[r] = np.nansum(np.digitize(sig_bin, wf_dege) == 1, axis = (0, 1))
    sig_in_wide[r] = np.nansum(np.digitize(sig_bin, wf_dege_wide) == 1, axis = (0, 1))
    signal_bin[r] = sig_bin
    del hf, wf_time, sig_bin, wf_dege, wf_dege_wide

    if Type == 'signal':
        hf_name = f'_AraOut.{Type}_E{int(exponent[r, 0])}_F{flavor[r]}_A{Station}_R{config[r]}.txt.run{sim_run[r]}.h5'
    if Type == 'noise':
        hf_name = f'_AraOut.{Type}_A{Station}_R{config[r]}.txt.run{sim_run[r]}.h5'
    try:
        hf = h5py.File(f'{q_path}qual_cut{hf_name}', 'r')
        qual_tot[r] = (hf['tot_qual_cut_sum'][:] != 0).astype(int)
        qual[r] = (hf['tot_qual_cut'][:] != 0).astype(int)
        evt_rate[r] = hf['evt_rate'][:]
        del hf
    except FileNotFoundError:
        print(f'{q_path}qual_cut{hf_name}')

    try:
        hf = h5py.File(f'{s_path}snr{hf_name}', 'r')
        snr_tot = hf['snr'][:]
        snr[r] = snr_tot
        ex_run = get_example_run(Station, config[r])
        bad_ant = known_issue.get_bad_antenna(ex_run)
        snr_tot[bad_ant] = np.nan
        snr_max[r, 0] = -np.sort(-snr_tot[:8], axis = 0)[2]
        snr_max[r, 1] = -np.sort(-snr_tot[8:], axis = 0)[2]
        del hf, snr_tot, ex_run, bad_ant
    except FileNotFoundError:
        print(f'{s_path}snr{hf_name}')

    try:
        hf = h5py.File(f'{r_path}reco_ele{hf_name}', 'r')
        coef_tot = hf['coef'][:] # pol, rad, sol, evt
        coord_tot = hf['coord'][:] # pol, tp, rad, sol, evt
        coef[r] = coef_tot
        coord[r] = coord_tot
        coef_re = np.reshape(coef_tot, (2, 4, -1))
        coord_re = np.reshape(coord_tot, (2, 2, 4, -1))
        coef_max_idx = np.nanargmax(coef_re, axis = 1)
        coef_max[r] = coef_re[pol_num[:, np.newaxis], coef_max_idx, evt_num[np.newaxis, :]]
        coord_max[r, :, :2] = coord_re[pol_num[:, np.newaxis, np.newaxis], pol_num[np.newaxis, :, np.newaxis], coef_max_idx, evt_num[np.newaxis, np.newaxis, :]]
        coord_max[r, :, 2] = rad_o[coef_max_idx // 2]
        coef_ele = hf['coef_ele'][:] # pol, theta, rad, sol, evt
        coef_ele_max = np.nanmax(coef_ele[:, sur_idx], axis = (1, 2, 3)) # pol, evt
        coef_ratio[r] = coef_ele_max / coef_max[r]
        del hf, coef_tot, coord_tot, coef_re, coord_re, coef_max_idx, coef_ele, coef_ele_max
    except FileNotFoundError:
        print(f'{r_path}reco_ele{hf_name}')

    try:
        hf = h5py.File(f'{c_path}csw{hf_name}', 'r')
        hill_max_idx[r] = hf['hill_max_idx'][:]
        hill_max[r] = hf['hill_max'][:]
        snr_csw[r] = hf['snr_csw'][:]
        cdf_avg[r] = hf['cdf_avg'][:]
        slope[r] = hf['slope'][:]
        intercept[r] = hf['intercept'][:]
        r_value[r] = hf['r_value'][:]
        p_value[r] = hf['p_value'][:]
        std_err[r] = hf['std_err'][:]
        ks[r] = hf['ks'][:]
        nan_flag[r] = hf['nan_flag'][:]
        del hf
    except FileNotFoundError:
        print(f'{c_path}csw{hf_name}')
    
    try:
        hf = h5py.File(f'{m_path}mf{hf_name}', 'r')
        mf_max[r] = hf['mf_max'][:] # pol, evt
        mf_temp = hf['mf_temp'][:, 1:3] # of pols, theta n phi, # of evts
        #mf_ser_max[r, :, 0] = theta_bin[mf_temp[:, 0]] # vh t
        mf_ser_max[r, :, 0] = mf_temp[:, 0] # vh t
        #mf_ser_max[r, :, 1] = phi_bin[mf_temp[:, 1]] # vh p
        mf_ser_max[r, :, 1] = mf_temp[:, 1] # vh p
        mf_max_each = hf['mf_max_each'][:]
        mf_the_max = np.nanmax(mf_max_each[:, :, :4], axis = (1, 2, 3))
        mf_ratio[r] = mf_the_max / mf_max[r]
        del hf, mf_temp, mf_max_each, mf_the_max
    except FileNotFoundError:
        print(f'{m_path}mf{hf_name}')
    del hf_name

path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/Hist/'
if not os.path.exists(path):
    os.makedirs(path)
os.chdir(path)

file_name = f'Sim_Summary_{Type}_A{Station}.h5'
hf = h5py.File(file_name, 'w')
hf.create_dataset('sim_run', data=sim_run, compression="gzip", compression_opts=9)
hf.create_dataset('config', data=config, compression="gzip", compression_opts=9)
hf.create_dataset('flavor', data=flavor, compression="gzip", compression_opts=9)
hf.create_dataset('radius', data=radius, compression="gzip", compression_opts=9)
hf.create_dataset('pnu', data=pnu, compression="gzip", compression_opts=9)
hf.create_dataset('inu_thrown', data=inu_thrown, compression="gzip", compression_opts=9)
hf.create_dataset('weight', data=weight, compression="gzip", compression_opts=9)
hf.create_dataset('probability', data=probability, compression="gzip", compression_opts=9)
hf.create_dataset('nuflavorint', data=nuflavorint, compression="gzip", compression_opts=9)
hf.create_dataset('nu_nubar', data=nu_nubar, compression="gzip", compression_opts=9)
hf.create_dataset('currentint', data=currentint, compression="gzip", compression_opts=9)
hf.create_dataset('elast_y', data=elast_y, compression="gzip", compression_opts=9)
hf.create_dataset('posnu', data=posnu, compression="gzip", compression_opts=9)
hf.create_dataset('posnu_antcen', data=posnu_antcen, compression="gzip", compression_opts=9)
hf.create_dataset('nnu', data=nnu, compression="gzip", compression_opts=9)
hf.create_dataset('rec_ang', data=rec_ang, compression="gzip", compression_opts=9)
hf.create_dataset('view_ang', data=view_ang, compression="gzip", compression_opts=9)
hf.create_dataset('arrival_time', data=arrival_time, compression="gzip", compression_opts=9)
hf.create_dataset('coef', data=coef, compression="gzip", compression_opts=9)
hf.create_dataset('coord', data=coord, compression="gzip", compression_opts=9)
hf.create_dataset('exponent', data=exponent, compression="gzip", compression_opts=9)
hf.create_dataset('evt_rate', data=evt_rate, compression="gzip", compression_opts=9)
hf.create_dataset('coef_max', data=coef_max, compression="gzip", compression_opts=9)
hf.create_dataset('coef_ratio', data=coef_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('coord_max', data=coord_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_max', data=mf_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_ser_max', data=mf_ser_max, compression="gzip", compression_opts=9)
hf.create_dataset('mf_ratio', data=mf_ratio, compression="gzip", compression_opts=9)
hf.create_dataset('snr', data=snr, compression="gzip", compression_opts=9)
hf.create_dataset('snr_max', data=snr_max, compression="gzip", compression_opts=9)
hf.create_dataset('qual', data=qual, compression="gzip", compression_opts=9)
hf.create_dataset('qual_tot', data=qual_tot, compression="gzip", compression_opts=9)
hf.create_dataset('sig_in', data=sig_in, compression="gzip", compression_opts=9)
hf.create_dataset('sig_in_wide', data=sig_in_wide, compression="gzip", compression_opts=9)
hf.create_dataset('signal_bin', data=signal_bin, compression="gzip", compression_opts=9)
hf.create_dataset('hill_max_idx', data=hill_max_idx, compression="gzip", compression_opts=9)
hf.create_dataset('hill_max', data=hill_max, compression="gzip", compression_opts=9)
hf.create_dataset('snr_csw', data=snr_csw, compression="gzip", compression_opts=9)
hf.create_dataset('cdf_avg', data=cdf_avg, compression="gzip", compression_opts=9)
hf.create_dataset('slope', data=slope, compression="gzip", compression_opts=9)
hf.create_dataset('intercept', data=intercept, compression="gzip", compression_opts=9)
hf.create_dataset('r_value', data=r_value, compression="gzip", compression_opts=9)
hf.create_dataset('p_value', data=p_value, compression="gzip", compression_opts=9)
hf.create_dataset('std_err', data=std_err, compression="gzip", compression_opts=9)
hf.create_dataset('ks', data=ks, compression="gzip", compression_opts=9)
hf.create_dataset('nan_flag', data=nan_flag, compression="gzip", compression_opts=9)
hf.create_dataset('z_bins', data=z_bins, compression="gzip", compression_opts=9)
hf.create_dataset('z_bin_center', data=z_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('a_bins', data=a_bins, compression="gzip", compression_opts=9)
hf.create_dataset('a_bin_center', data=a_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('c_bins', data=c_bins, compression="gzip", compression_opts=9)
hf.create_dataset('c_bin_center', data=c_bin_center, compression="gzip", compression_opts=9)
hf.create_dataset('m_bins', data=m_bins, compression="gzip", compression_opts=9)
hf.create_dataset('m_bin_center', data=m_bin_center, compression="gzip", compression_opts=9)
hf.close()
print('file is in:',path+file_name, size_checker(path+file_name))




