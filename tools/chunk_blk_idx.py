import numpy as np
from tqdm import tqdm
import h5py

def blk_idx_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting block index starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_quality_cut import qual_cut_loader
    from tools.ara_run_manager import run_info_loader
    from tools.ara_wf_analyzer import hist_loader

    # geom. info.
    ara_const = ara_const()
    num_blks = ara_const.BLOCKS_PER_DDA
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    unix_time = ara_uproot.unix_time
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    daq_qual_sum = np.nansum(total_qual_cut[:, :6], axis = 1)
    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 0)
    clean_evt = evt_num[clean_evt_idx]
    print(f'Number of clean event is {len(clean_evt)}')
    del qual_cut_sum, ara_qual

    # ped info
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = True)
    ped_dat = run_info.get_result_path(file_type = 'ped', verbose = True)
    ped_hf = h5py.File(ped_dat, 'r')
    ped_qualities = ped_hf['ped_qualities'][:]
    del run_info, ped_dat, ped_hf

    # output array
    blk_len = 50
    blk_idx = np.full((blk_len, num_evts), np.nan, dtype = float)   
    del blk_len

    # loop over the events
    for evt in tqdm(range(num_evts)):

        if daq_qual_sum[evt] != 0:
            continue

        blk_idx_arr, blk_idx_len = ara_uproot.get_block_idx(evt, trim_1st_blk = True)
        blk_idx[:blk_idx_len, evt] = blk_idx_arr
        del blk_idx_arr, blk_idx_len
    del ara_uproot, num_evts, daq_qual_sum

    blk_idx_rf = np.copy(blk_idx)
    blk_idx_rf[:, trig_type != 0] = np.nan
    blk_idx_cal = np.copy(blk_idx)
    blk_idx_cal[:, trig_type != 1] = np.nan
    blk_idx_soft = np.copy(blk_idx)
    blk_idx_soft[:, trig_type != 2] = np.nan
    blk_idx_clean = np.copy(blk_idx)
    blk_idx_clean[:,  ~clean_evt_idx] = np.nan
    blk_idx_ped = np.copy(blk_idx)
    #blk_idx_ped[:, ped_qualities == 0] = np.nan

    blk_range = np.arange(num_blks)
    blk_bins = np.linspace(-0.5, num_blks - 0.5, num_blks + 1)
    blk_bin_center = (blk_bins[1:] + blk_bins[:-1]) / 2
    blk_hist = np.histogram(blk_idx.flatten(), bins = blk_bins)[0].astype(int) 
    blk_rf_hist = np.histogram(blk_idx_rf.flatten(), bins = blk_bins)[0].astype(int) 
    blk_cal_hist = np.histogram(blk_idx_cal.flatten(), bins = blk_bins)[0].astype(int) 
    blk_soft_hist = np.histogram(blk_idx_soft.flatten(), bins = blk_bins)[0].astype(int) 
    blk_clean_hist = np.histogram(blk_idx_clean.flatten(), bins = blk_bins)[0].astype(int) 
    blk_ped_hist = np.histogram(blk_idx_ped.flatten(), bins = blk_bins)[0].astype(int) 
    del num_blks, clean_evt_idx 

    print('Block index collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'total_qual_cut':total_qual_cut,
            'clean_evt':clean_evt,
            'ped_qualities':ped_qualities,
            'blk_idx':blk_idx,
            'blk_idx_rf':blk_idx_rf,
            'blk_idx_cal':blk_idx_cal,
            'blk_idx_soft':blk_idx_soft,
            'blk_idx_clean':blk_idx_clean,
            'blk_idx_ped':blk_idx_ped,
            'blk_range':blk_range,
            'blk_bins':blk_bins,
            'blk_bin_center':blk_bin_center,
            'blk_hist':blk_hist,
            'blk_rf_hist':blk_rf_hist,
            'blk_cal_hist':blk_cal_hist,
            'blk_soft_hist':blk_soft_hist,
            'blk_clean_hist':blk_clean_hist,
            'blk_ped_hist':blk_ped_hist}




