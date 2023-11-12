import numpy as np
from tqdm import tqdm

def mf_s1_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting mf second half starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_matched_filter import ara_matched_filter  
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_quality_cut import get_bad_events

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    if use_l2:
        ara_root = ara_l2_loader(Data)
        num_evts = ara_root.num_evts
        evt_num = ara_root.evt_num
        daq_qual_cut_sum = ara_root.daq_cut
        trig_type = ara_root.trig_type
        st = ara_root.station_id
        yr = ara_root.year
        run = ara_root.run
    else:
        ara_uproot = ara_uproot_loader(Data)
        evt_num = ara_uproot.evt_num
        num_evts = ara_uproot.num_evts
        trig_type = ara_uproot.get_trig_type()
        st = ara_uproot.station_id
        yr = ara_uproot.year
        run = ara_uproot.run
        ara_root = ara_root_loader(Data, Ped, st, yr)
        del ara_uproot
    del yr

    # pre quality cut
    if use_l2 == False:
        daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, qual_type = 2)[0]

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    good_ant_idx = bad_ant == 0
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # matched filter
    ara_mf = ara_matched_filter(st, run, wf_int.dt, wf_int.pad_len, get_sub_file = True, verbose = True)  
    good_ch_len = ara_mf.good_ch_len
    num_temp_params = ara_mf.num_temp_params
    num_arr_params = ara_mf.num_arr_params
    mf_param_shape = ara_mf.mf_param_shape
    mf_param_com_shape = ara_mf.mf_param_com_shape
    num_pols = ara_mf.num_pols
    num_pols_com = ara_mf.num_pols_com
    del st, run
     
    mf_max = np.full((num_pols_com, num_evts), np.nan, dtype = float) # array dim: (# of pols, # of evts)
    mf_max_each = np.full((num_pols_com, num_temp_params[0], num_arr_params[0], num_arr_params[1], num_evts), np.nan, dtype = float) # array dim: (# of pols, # of shos, # of thetas, # of phis, # of evts)
    mf_temp = np.full((num_pols, mf_param_shape[1], num_evts), -1, dtype = int) # array dim: (# of pols, # of temp params (sho, theta, phi, off (8)), # of evts) 
    mf_temp_com = np.full((mf_param_com_shape, num_evts), -1, dtype = int) # array dim: (# of temp params (sho, theta, phi, off (16)), # of evts) 
    mf_temp_off = np.full((good_ch_len, num_temp_params[0], num_temp_params[1], num_evts), np.nan, dtype = float) #  arr dim: (# of good ants, # of shos, # of ress)
    mf_indi = np.full((num_ants, num_temp_params[0], num_temp_params[1], num_temp_params[2], num_evts), np.nan, dtype = float) # chs, shos, ress, offs, evts
    del num_pols, num_pols_com, mf_param_shape, mf_param_com_shape, good_ch_len, num_temp_params, num_arr_params

    half_num_evts = num_evts // 2
    half_num_evts1 = int(half_num_evts - 1)
    sel_evt_idx = (np.arange(num_evts, dtype = int) > half_num_evts1).astype(int)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:
      if evt > half_num_evts1:

        if daq_qual_cut_sum[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_nan_pad = True, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_mf.get_evt_wise_snr(wf_int.pad_v) 
        mf_max[:, evt] = ara_mf.mf_max
        mf_max_each[:, :, :, :, evt] = ara_mf.mf_max_each
        mf_temp[:, :, evt] = ara_mf.mf_temp
        mf_temp_com[:, evt] = ara_mf.mf_temp_com
        mf_temp_off[:, :, :, evt] = ara_mf.mf_temp_off
        mf_indi[good_ant_idx, :, :, :, evt] = ara_mf.corr_max
        #print(mf_max[:, evt], mf_temp[:, :, evt])
    del ara_root, num_evts, num_ants, wf_int, ara_mf, daq_qual_cut_sum, good_ant_idx

    print('MF second half collecting is done!')
    
    return {'evt_num':evt_num,
            'sel_evt_idx':sel_evt_idx,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'mf_max':mf_max,
            'mf_max_each':mf_max_each,
            'mf_temp':mf_temp,
            'mf_temp_com':mf_temp_com,
            'mf_temp_off':mf_temp_off,
            'mf_indi':mf_indi}







