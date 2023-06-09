import os
import numpy as np
from tqdm import tqdm
import h5py

def qual_cut_3rd_collector(Data, st, run, qual_type = 3, analyze_blind_dat = False, no_tqdm = False):

    print('Quality cut 3rd starts!')

    #from tools.ara_constant import ara_const   
    from tools.ara_run_manager import run_info_loader
    from tools.ara_quality_cut import filt_qual_cut_loader
    from tools.ara_quality_cut import run_qual_cut_loader
    from tools.ara_quality_cut import get_bad_live_time

    ## load results
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    qual_dat = run_info.get_result_path(file_type = 'qual_cut_2nd', verbose = True)
    qual_hf = h5py.File(qual_dat, 'r')    
    evt_num = qual_hf['evt_num'][:]
    num_evts = len(evt_num)
    entry_num = qual_hf['entry_num'][:]
    trig_type = qual_hf['trig_type'][:]
    unix_time = qual_hf['unix_time'][:]
    pps_number = qual_hf['pps_number'][:]
    time_bins_sec = qual_hf['time_bins_sec'][:]
    time_bins = qual_hf['time_bins'][:]
    sec_per_sec = qual_hf['sec_per_sec'][:]
    sec_per_min = qual_hf['sec_per_min'][:]
    pre_qual_cut = qual_hf['pre_qual_cut'][:]
    pre_qual_cut_sum = qual_hf['pre_qual_cut_sum'][:]
    daq_qual_cut_sum = qual_hf['daq_qual_cut_sum'][:]
    ped_qual_cut = qual_hf['ped_qual_cut'][:]
    ped_qual_cut_sum = qual_hf['ped_qual_cut_sum'][:]
    tot_qual_cut = qual_hf['tot_qual_cut'][:]
    #tot_qual_cut_sum = qual_hf['tot_qual_cut_sum'][:]
    ped_qual_evt_num = qual_hf['ped_qual_evt_num'][:]
    ped_qual_type = qual_hf['ped_qual_type'][:]
    ped_qual_num_evts = qual_hf['ped_qual_num_evts'][:]
    ped_blk_usage = qual_hf['ped_blk_usage'][:]
    ped_low_blk_usage = qual_hf['ped_low_blk_usage'][:]
    ped_qualities = qual_hf['ped_qualities'][:]
    ped_counts = qual_hf['ped_counts'][:]
    ped_final_type = qual_hf['ped_final_type'][:]
    #bad_run = qual_hf['bad_run'][:]
    #tot_qual_live_time = qual_hf['tot_qual_live_time'][:]
    #tot_qual_bad_live_time = qual_hf['tot_qual_bad_live_time'][:]
    #tot_qual_sum_bad_live_time = qual_hf['tot_qual_sum_bad_live_time'][:]
    del run_info, qual_dat, qual_hf

    ## filter cut
    filt_qual = filt_qual_cut_loader(st, run, evt_num, analyze_blind_dat = analyze_blind_dat, verbose = True, spark_unblind = False, cal_sur_unblind = True) 
    filt_qual_cut = filt_qual.run_filt_qual_cut()
    filt_qual_cut_sum = filt_qual.filt_qual_cut_sum
    del filt_qual

    # total quality cut
    tot_qual_cut = np.append(tot_qual_cut, filt_qual_cut, axis = 1)
    tot_qual_cut_copy = np.copy(tot_qual_cut)
    tot_qual_cut_copy[:, 14] = 0
    tot_qual_cut_sum = np.nansum(tot_qual_cut_copy, axis = 1)
    del tot_qual_cut_copy

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, tot_qual_cut, analyze_blind_dat = analyze_blind_dat, qual_type = qual_type, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual

    # live time
    tot_qual_live_time, tot_qual_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, tot_qual_cut, verbose = True)
    tot_qual_sum_bad_live_time = get_bad_live_time(trig_type, unix_time, time_bins_sec, sec_per_sec, np.nansum(tot_qual_cut, axis = 1), verbose = True)[1]

    print('Quality cut 3rd is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'time_bins_sec':time_bins_sec,
            'time_bins':time_bins,
            'sec_per_sec':sec_per_sec,
            'sec_per_min':sec_per_min,
            'pre_qual_cut':pre_qual_cut,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'ped_qual_cut':ped_qual_cut,
            'ped_qual_cut_sum':ped_qual_cut_sum,
            'filt_qual_cut':filt_qual_cut,
            'filt_qual_cut_sum':filt_qual_cut_sum,
            'tot_qual_cut':tot_qual_cut,
            'tot_qual_cut_sum':tot_qual_cut_sum,
            'ped_qual_evt_num':ped_qual_evt_num,
            'ped_qual_type':ped_qual_type,
            'ped_qual_num_evts':ped_qual_num_evts,
            'ped_blk_usage':ped_blk_usage,
            'ped_low_blk_usage':ped_low_blk_usage,
            'ped_qualities':ped_qualities,
            'ped_counts':ped_counts,
            'ped_final_type':ped_final_type,
            'bad_run':bad_run,
            'tot_qual_live_time':tot_qual_live_time,
            'tot_qual_bad_live_time':tot_qual_bad_live_time,
            'tot_qual_sum_bad_live_time':tot_qual_sum_bad_live_time}





