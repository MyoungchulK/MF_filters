import os
import numpy as np
from tqdm import tqdm
import h5py

def qual_cut_3rd_collector(Data, st, run, qual_type = 3, analyze_blind_dat = False, no_tqdm = False):

    print('Quality cut 3rd starts!')

    from tools.ara_constant import ara_const   
    from tools.ara_run_manager import run_info_loader
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_quality_cut import get_calpulser_cut
    from tools.ara_quality_cut import run_qual_cut_loader
    from tools.ara_quality_cut import get_bad_live_time
    from tools.ara_quality_cut import quick_qual_check

    ara_const = ara_const()
    num_sts = ara_const.DDA_PER_ATRI
    del ara_const

    ## temp ##
    force_ub = True

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
    del qual_dat, qual_hf

    ## filter cut
    filt_qual_cut = np.full((num_evts, 3), 0, dtype = int)

    ## cal & surface cut
    reco_dat = run_info.get_result_path(file_type = 'reco', verbose = True, force_unblind = force_ub, return_none = True)
    if reco_dat is None:
        filt_qual_cut[:, 0] = 0 
        filt_qual_cut[:, 1] = 0 
    else:
        reco_hf = h5py.File(reco_dat, 'r')
        evt_num_reco = reco_hf['evt_num'][:]
        num_evts_reco = len(evt_num_reco)
        coord_max = reco_hf['coord'][:]
        trig_type_reco = reco_hf['trig_type'][:]
        del reco_dat, reco_hf

        cp_cut, num_cuts, pol_idx = get_calpulser_cut(st, run)
        cal_cuts = np.full((num_evts_reco), 0, dtype = int)
        for cal in range(num_cuts):
            ele_flag = np.digitize(coord_max[pol_idx, 0, 0, 0], cp_cut[cal, 0]) == 1
            azi_flag = np.digitize(coord_max[pol_idx, 1, 0, 0], cp_cut[cal, 1]) == 1
            cal_cuts += np.logical_and(ele_flag, azi_flag).astype(int)
            del ele_flag, azi_flag
        cal_cuts[trig_type_reco == 1] = 0
        del cp_cut, num_cuts, pol_idx, trig_type_reco

        sr_cut = np.array([35], dtype = float)
        coord_max_flat = np.reshape(coord_max[:, 0, 1, :, :], (4, -1))
        sur_cuts = np.any(coord_max_flat > sr_cut, axis = 0).astype(int) 

        if num_evts != num_evts_reco:
            evt_idx = np.in1d(evt_num, evt_num_reco)
        else:
            evt_idx = np.full((num_evts), True, dtype = bool)
        try:
            filt_qual_cut[evt_idx, 0] = cal_cuts
            filt_qual_cut[evt_idx, 1] = sur_cuts
        except ValueError:
            for evt in tqdm(range(num_evts_reco)):
                evt_idx = np.where(evt_num == evt_num_reco[evt])[0]
                if len(evt_idx) > 0:
                    filt_qual_cut[evt_idx, 0] = cal_cuts[evt]
                    filt_qual_cut[evt_idx, 1] = sur_cuts[evt]
        del coord_max, evt_num_reco, num_evts_reco, coord_max_flat, sr_cut, evt_idx

    filt_qual_cut[:, 2] = 0
    snr_dat = run_info.get_result_path(file_type = 'snr', verbose = True, force_unblind = force_ub, return_none = True)
    if snr_dat is None:
        filt_qual_cut[:, 2] = 0 
    else:
        ## over powered string
        known_issue = known_issue_loader(st, verbose = True)
        bad_ant = known_issue.get_bad_antenna(run)
        del known_issue

        snr_hf = h5py.File(snr_dat, 'r')
        evt_num_snr = snr_hf['evt_num'][:]
        num_evts_snr = len(evt_num_snr)
        trig_type_snr = snr_hf['trig_type'][:]
        rms = snr_hf['rms'][:]
        rms[bad_ant] = np.nan
        rms[:, trig_type_snr == 1] = np.nan
        del snr_dat, snr_hf, bad_ant, trig_type_snr

        pow_n = rms ** 2
        pow_n_avg = np.full((num_sts, num_evts_snr), np.nan, dtype = float)
        for st in range(num_sts):
            pow_n_avg[st] = np.nanmean(pow_n[st::num_sts], axis = 0)
        pow_n_avg_sort = -np.sort(-pow_n_avg, axis = 0)
        pow_ratio = pow_n_avg_sort[0] / pow_n_avg_sort[1]
        ratio_cut = np.array([5], dtype = float)
        op_cuts = (pow_ratio > ratio_cut).astype(int)

        if num_evts != num_evts_snr:
            evt_idx = np.in1d(evt_num, evt_num_snr)
        else:
            evt_idx = np.full((num_evts), True, dtype = bool)
        try:
            filt_qual_cut[evt_idx, 2] = op_cuts
        except ValueError:
            for evt in tqdm(range(num_evts_snr)):
                evt_idx = np.where(evt_num == evt_num_snr[evt])[0]
                if len(evt_idx) > 0:
                    filt_qual_cut[evt_idx, 2] = op_cuts[evt]
        del op_cuts, pow_n_avg_sort, pow_n_avg, pow_n, rms, pow_ratio, ratio_cut, num_evts_snr, evt_num_snr, evt_idx
    del num_sts, run_info, num_evts

    filt_qual_cut_sum = np.nansum(filt_qual_cut, axis = 1)
    quick_qual_check(filt_qual_cut[:, 0] != 0, 'calpulser cut', evt_num)
    quick_qual_check(filt_qual_cut[:, 1] != 0, 'surface cut', evt_num)
    quick_qual_check(filt_qual_cut[:, 2] != 0, 'op antenna cut', evt_num)
    quick_qual_check(filt_qual_cut_sum != 0, 'total filter qual cut!', evt_num)

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





