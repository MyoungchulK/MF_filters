import numpy as np
from tqdm import tqdm
import h5py

def cw_add_collector(Station, Run, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_dat = run_info.get_result_path(file_type = 'cw', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')

    rf_evt = cw_hf['rf_evt'][:]
    clean_evt = cw_hf['clean_evt'][:]
    clean_idx = np.in1d(rf_evt, clean_evt)
    del rf_evt, clean_evt

    print('data loading')
    amp_err_bins = cw_hf['amp_err_bins'][:]
    phase_err_bins = cw_hf['phase_err_bins'][:]
    sub_amp_err = cw_hf['sub_amp_err'][:]
    sub_phase_err = cw_hf['sub_phase_err'][:]

    print('hist making')
    ara_hist = hist_loader(amp_err_bins, phase_err_bins)
    amp_err_phase_err_rf_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, use_flat = True)
    amp_err_phase_err_rf_cut_map = ara_hist.get_2d_hist(sub_amp_err, sub_phase_err, cut = ~clean_idx, use_flat = True)
    del ara_hist, cw_hf, clean_idx, amp_err_bins, phase_err_bins, sub_amp_err, sub_phase_err
    print(amp_err_phase_err_rf_map.shape)
    print(amp_err_phase_err_rf_cut_map.shape)

    hf = h5py.File(cw_dat, 'a')   
    hf.create_dataset('amp_err_phase_err_rf_map', data=amp_err_phase_err_rf_map, compression="gzip", compression_opts=9)
    hf.create_dataset('amp_err_phase_err_rf_cut_map', data=amp_err_phase_err_rf_cut_map, compression="gzip", compression_opts=9)
    hf.close() 
    del hf, amp_err_phase_err_rf_map, amp_err_phase_err_rf_cut_map

    hf = h5py.File(cw_dat, 'r')
    for f in list(hf):
        print(f)
    del hf, run_info, cw_dat

    print('cw collecting is done!')

    return 

