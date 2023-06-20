import numpy as np
from tqdm import tqdm

def csw_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting csw starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_csw import ara_csw
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_quality_cut import get_bad_events

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_pols = ara_const.POLARIZATION
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
    del known_issue

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # csw
    ara_csw = ara_csw(st, run, wf_int.dt, wf_int.pad_zero_t, get_sub_file = True, verbose = True)
    num_sols = ara_csw.num_sols
    del st, run
    
    # output array
    hill_max_idx = np.full((num_pols, num_sols, num_evts), np.nan, dtype = float)
    hill_max = np.copy(hill_max_idx)
    snr_csw = np.copy(hill_max_idx)
    cdf_avg = np.copy(hill_max_idx) 
    slope = np.copy(hill_max_idx)
    intercept = np.copy(hill_max_idx)
    r_value = np.copy(hill_max_idx)
    p_value = np.copy(hill_max_idx)
    std_err = np.copy(hill_max_idx)
    ks = np.copy(hill_max_idx)
    nan_flag = np.full((num_pols, num_sols, num_evts), 0, dtype = int)
    del num_sols, num_pols

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:

        if daq_qual_cut_sum[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)

        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = False, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        ara_csw.get_csw_params(wf_int.pad_t, wf_int.pad_v, wf_int.pad_num, evt)
        hill_max_idx[:, :, evt] = ara_csw.hill_max_idx
        hill_max[:, :, evt] = ara_csw.hill_max
        snr_csw[:, :, evt] = ara_csw.snr_csw
        cdf_avg[:, :, evt] = ara_csw.cdf_avg
        slope[:, :, evt] = ara_csw.slope
        intercept[:, :, evt] = ara_csw.intercept
        r_value[:, :, evt] = ara_csw.r_value
        p_value[:, :, evt] = ara_csw.p_value
        std_err[:, :, evt] = ara_csw.std_err
        ks[:, :, evt] = ara_csw.ks
        nan_flag[:, :, evt] = ara_csw.nan_flag
        if np.any(nan_flag[:, :, evt]):
            print(hill_max_idx[:, :, evt], hill_max[:, :, evt], snr_csw[:, :, evt], cdf_avg[:, :, evt], slope[:, :, evt], intercept[:, :, evt], r_value[:, :, evt], p_value[:, :, evt], std_err[:, :, evt], ks[:, :, evt]) # debug
    del ara_root, num_evts, num_ants, wf_int, ara_csw, daq_qual_cut_sum

    print('CSW collecting is done!')
    
    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'hill_max_idx':hill_max_idx,
            'hill_max':hill_max,
            'snr_csw':snr_csw,
            'cdf_avg':cdf_avg,
            'slope':slope,
            'intercept':intercept,
            'r_value':r_value,
            'p_value':p_value,
            'std_err':std_err,
            'ks':ks,
            'nan_flag':nan_flag}






