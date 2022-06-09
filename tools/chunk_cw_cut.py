import numpy as np
from tqdm import tqdm

def cw_cut_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting cw starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_run_manager import run_info_loader
    from tools.ara_quality_cut import cw_qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION 
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year) 
    knwon_issue = known_issue_loader(st)
    bad_ant = knwon_issue.get_bad_antenna(run) == 1
    del knwon_issue, ara_uproot

    # daq qulity cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_cut_dat = run_info.get_result_path(file_type = 'daq_cut', verbose = True)
    daq_cut_hf = h5py.File(daq_cut_dat, 'r')
    daq_cut = daq_cut_hf['total_daq_cut'][:]
    daq_cut[:, 10] = 0 # disable bad unix time
    daq_cut[:, 21] = 0 # disable known bad run
    daq_cut_sum = np.nansum(daq_cut, axis = 1)
    clean_evt_idx = np.logical(trig_type == 0, daq_cut_sum == 0)
    clean_entry = entry_num[clean_evt_idx]
    clean_evt = evt_num[clean_evt_idx]
    num_clean_evts = len(clean_evt)
    print(f'Number of clean event is {num_clean_evts}') 
    del clean_evt_idx run_info, daq_cut_dat, daq_cut_hf, daq_cut

    # cw quality cut
    cw_qual = cw_qual_cut_loader(st, run, evt_num, pps_number, verbose = True)
    cw_params = cw_qual.ratio_cut
    del st, run

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, cw_params = cw_params)

    # output
    sol_pad = 50
    sub_freq = np.full((sol_pad, num_ants, num_clean_evts), np.nan, dtype = float)
    sub_amp_err = np.copy(sub_freq)
    sub_amp_err[0, ~bad_ant] = 0
    sub_ratio = np.copy(sub_freq)
    rp_evt_num = []
    rp_entry_num = []
    rp_wf = []
    del sol_pad

    # loop over the events
    for evt in tqdm(range(num_clean_evts)):
      #if evt == 100:        

        # get entry and wf
        ara_root.get_entry(clean_entry[evt])
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        sub_ant_counts = 0
        for ant in range(num_ants):
            if bad_ant[ant]:
                continue                
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True)
            num_sols = wf_int.sin_sub.num_sols
            if num_sols:
                sub_ant_counts += 1 
                num_sols += 1
                sub_freq[1:num_sols, ant, evt] = wf_int.sin_sub.sub_freqs
                sub_amp_err[1:num_sols, ant, evt] = wf_int.sin_sub.sub_amp_errs
                sub_ratio[:num_sols, ant, evt] = wf_int.sin_sub.sub_ratios
            del raw_t, raw_v, num_sols
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()

        # cw check
        if cw_qual.run_cw_qual_cut(clean_entry[evt], sub_ant_counts):
            rp_evt_num.append(clean_evt[evt])
            rp_entry_num.append(clean_entry[evt])
            rp_wfs = wf_int.pad_v
            rp_wfs[:, bad_ant] = np.nan
            rp_wf.append(rp_wfs)
        del sub_ant_counts
    del ara_root, num_clean_evts, num_ants, wf_int

    # quality output
    total_cw_cut = cw_qual.get_cw_qual_cut()
    total_cw_cut_sum = cw_qual.cw_qual_cut_sum
    rp_evt_idx = cw_qual.rp_evts
    del cw_qual

    # to numpy array
    rp_evt_num = np.asarray(rp_evt_num).astype(int)
    rp_entry_num = np.asarray(rp_entry_num).astype(int)
    rp_wf = np.asarray(rp_wf)
    rp_cw_idx = ~np.in1d(rp_evt_num, evt_num[total_cw_cut_sum != 0])
    rp_evt_num = rp_evt_num[rp_cw_idx]
    rp_entry_num = rp_entry_num[rp_cw_idx]
    rp_wf = rp_wf[rp_cw_idx]
    del rp_cw_idx

    print('cw collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'bad_ant':bad_ant.astype(int),
            'daq_cut_sum':daq_cut_sum,
            'clean_evt':clean_evt,
            'clean_entry':clean_entry,
            'cw_params':cw_params,
            'total_cw_cut':total_cw_cut,
            'total_cw_cut_sum':total_cw_cut_sum,
            'sub_freq':sub_freq,
            'sub_amp_err':sub_amp_err,
            'sub_ratio':sub_ratio,
            'rp_evt_idx':rp_evt_idx,
            'rp_evt_num':rp_evt_num,
            'rp_entry_num':rp_entry_num,
            'rp_wf':rp_wf}
