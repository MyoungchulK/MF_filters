import os
import numpy as np
from tqdm import tqdm

def qual_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Quality cut starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_run_manager import run_info_loader
    from tools.ara_quality_cut import run_qual_cut_loader
    from tools.ara_quality_cut import get_live_time

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    trig_type = ara_uproot.get_trig_type()
    pps_number = ara_uproot.pps_number
    unix_time = ara_uproot.unix_time
    st = ara_uproot.station_id
    run = ara_uproot.run

    # all quality cuts
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    del ara_uproot    

    daq_dat = run_info.get_result_path(file_type = 'daq_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    daq_cut = daq_hf['total_daq_cut'][:]
    total_daq_cut_sum = daq_hf['total_daq_cut_sum'][:]
    daq_qual_cut_sum = daq_hf['daq_qual_cut_sum'][:]
    pre_qual_cut_sum = daq_hf['pre_qual_cut_sum'][:]
    post_qual_cut_sum = daq_hf['post_qual_cut_sum'][:]
    del daq_dat, daq_hf    

    cw_dat = run_info.get_result_path(file_type = 'cw_cut', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    cw_cut = daq_hf['total_cw_cut'][:]
    total_cw_cut_sum = daq_hf['total_cw_cut_sum'][:]
    del cw_dat, cw_hf

    ped_dat = run_info.get_result_path(file_type = 'ped_cut', verbose = True)
    ped_hf = h5py.File(ped_dat, 'r')
    ped_cut = ped_hf['total_ped_cut'][:]
    total_ped_cut_sum = ped_hf['total_ped_cut_sum'][:]
    del ped_dat, ped_hf, run_info

    # total quality cut
    total_qual_cut = np.append(daq_cut, cw_cut, axis = 1)
    total_qual_cut = np.append(total_qual_cut, ped_cut, axis = 1)
    total_qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del daq_cut, cw_cut, ped_cut

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, total_qual_cut, analyze_blind_dat = analyze_blind_dat, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual

    # event_number
    rf_evt_num = evt_num[trig_type == 0]
    clean_evt_num = evt_num[tot_cut_sum == 0]
    clean_rf_evt_num = evt_num[(tot_cut_sum == 0) & (trig_type == 0)]
    rf_entry_num = entry_num[trig_type == 0]
    clean_entry_num = entry_num[tot_cut_sum == 0]
    clean_rf_entry_num = entry_num[(tot_cut_sum == 0) & (trig_type == 0)]

    # live time
    live_time, clean_live_time = get_live_time(st, run, unix_time, cut = tot_cut_sum, use_dead = True, verbose = True)
    del st, run

    print('Quality cut is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'rf_evt_num':rf_evt_num,
            'clean_evt_num':clean_evt_num,
            'clean_rf_evt_num':clean_rf_evt_num,
            'rf_entry_num':rf_entry_num,
            'clean_entry_num':clean_entry_num,
            'clean_rf_entry_num':clean_rf_entry_num,
            'total_qual_cut':total_qual_cut,
            'total_qual_cut_sum':total_qual_cut_sum,
            'total_daq_cut_sum':total_daq_cut_sum,
            'total_cw_cut_sum':total_cw_cut_sum,
            'total_ped_cut_sum':total_ped_cut_sum,
            'daq_qual_cut_sum':daq_qual_cut_sum,
            'pre_qual_cut_sum':pre_qual_cut_sum,
            'post_qual_cut_sum':post_qual_cut_sum,
            'bad_run':bad_run,
            'live_time':live_time,
            'clean_live_time':clean_live_time}

