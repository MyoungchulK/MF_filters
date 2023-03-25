import os
import h5py
import numpy as np
from tqdm import tqdm

def cw_flag_st_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting cw flag string starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_cw_filters_st import py_testbed
    from tools.ara_quality_cut import get_bad_events
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_utility import size_checker

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    trig_type = ara_uproot.get_trig_type()
    evt_num = ara_uproot.evt_num
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    yr = ara_uproot.year
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, yr)
    del ara_uproot, yr

    # pre quality cut
    daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, use_1st = True)[0]

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_freq_pad = True, use_rfft = True)
    freq_range = wf_int.pad_zero_freq

    # cw class
    cw_testbed = py_testbed(st, run, freq_range, analyze_blind_dat = analyze_blind_dat, verbose = True)
    testbed_params = np.array([cw_testbed.dB_cut, cw_testbed.dB_cut_broad, cw_testbed.num_coinc, cw_testbed.freq_range_broad, cw_testbed.freq_range_near])
    del st, run

    # output array  
    testbed_idx = []
    empty = np.full((0), 0, dtype = int)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
 
        if daq_qual_cut_sum[evt]:
            testbed_idx.append(empty)
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
    
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        wf_int.get_fft_wf(use_zero_pad = True, use_rfft = True, use_abs = True, use_norm = True, use_dBmHz = True)
        rfft_dbmhz = wf_int.pad_fft
        cw_testbed.get_bad_magnitude(rfft_dbmhz, trig_type[evt])
        testbed_idxs = cw_testbed.bad_idx
        testbed_idx.append(testbed_idxs)
        del rfft_dbmhz
    del ara_root, num_ants, wf_int, cw_testbed, daq_qual_cut_sum

    # to numpy array
    testbed_idx = np.asarray(testbed_idx)

    print('CW flag string collecting is done!')

    return {'evt_num':evt_num,
            'bad_ant':bad_ant,
            'freq_range':freq_range,
            'testbed_idx':testbed_idx,
            'testbed_params':testbed_params}









