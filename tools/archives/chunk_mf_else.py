import os
import numpy as np
from tqdm import tqdm
import h5py

def mf_else_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting mf starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_run_manager import run_info_loader
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_matched_filter import ara_matched_filter

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    num_evts = ara_uproot.num_evts
    trig_type = ara_uproot.get_trig_type()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)

    known_issue = known_issue_loader(ara_uproot.station_id)
    bad_ant = known_issue.get_bad_antenna(ara_uproot.run)
    del known_issue

    # pre quality cut
    run_info = run_info_loader(ara_uproot.station_id, ara_uproot.run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    tot_qual_cut_sum = daq_hf['tot_qual_cut_sum'][:]
    cw_dat = run_info.get_result_path(file_type = 'cw_cut', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    cw_qual_cut_sum = cw_hf['cw_qual_cut_sum'][:]
    tot_qual_cut_sum += cw_qual_cut_sum
    tot_qual_cut_sum = tot_qual_cut_sum.astype(int)
    del daq_dat, daq_hf, cw_dat, cw_hf, cw_qual_cut_sum

    # snr info
    snr_dat = run_info.get_result_path(file_type = 'snr', verbose = True)
    snr_hf = h5py.File(snr_dat, 'r')
    snr_weights = snr_hf['snr'][:]
    snr_copy = np.copy(snr_weights)
    snr_copy[bad_ant] = np.nan
    v_sum = np.nansum(snr_copy[:8], axis = 0)
    h_sum = np.nansum(snr_copy[8:], axis = 0)
    snr_weights[:8] /= v_sum
    snr_weights[8:] /= h_sum
    del snr_copy, v_sum, h_sum, snr_dat, snr_hf

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)
    dt = wf_int.dt
    wf_len = wf_int.pad_len

    ara_mf = ara_matched_filter(ara_uproot.station_id, ara_uproot.run, dt, wf_len, get_sub_file = True)
    del bad_ant, run_info, dt, wf_len, ara_uproot    

    evt_wise = np.full((2, num_evts), np.nan, dtype = float)
    evt_wise_ant = np.full((num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if tot_qual_cut_sum[evt]:
            continue
        if trig_type[evt] == 0:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        evt_wise[:, evt], evt_wise_ant[:, evt] = ara_mf.get_evt_wise_snr(wf_int.pad_v, wf_int.pad_num, snr_weights[:, evt]) 
       
    del trig_type, ara_root, num_evts, num_ants, wf_int, ara_mf, tot_qual_cut_sum, snr_weights

    print('MF collecting is done!')

    return {'evt_wise':evt_wise,
            'evt_wise_ant':evt_wise_ant}









