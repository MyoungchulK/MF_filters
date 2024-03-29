import os
import numpy as np
from tqdm import tqdm
import h5py

def mf_collector(Data, Ped, analyze_blind_dat = False):

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
    evt_num = ara_uproot.evt_num
    entry_num = ara_uproot.entry_num
    unix_time = ara_uproot.unix_time
    pps_number = ara_uproot.pps_number
    trig_type = ara_uproot.get_trig_type()
    num_evts = ara_uproot.num_evts
    st = ara_uproot.station_id
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, ara_uproot.year)
    del ara_uproot

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run)
    del known_issue

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    daq_qual_cut_sum = daq_hf['daq_qual_cut_sum'][:]
    del daq_dat, daq_hf

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
    del run_info, snr_dat, snr_hf, snr_copy, v_sum, h_sum

    # rayl table check
    bad_path = f'../data/rayl_runs/rayl_run_A{st}.txt'
    bad_run_arr = []
    with open(bad_path, 'r') as f:
        for lines in f:
            run_num = int(lines)
            bad_run_arr.append(run_num)
    bad_run_arr = np.asarray(bad_run_arr, dtype = int)
    if run in bad_run_arr:
        print(f'Bad noise modeling for A{st} R{run}! So, no MF results!')
        evt_wise = np.full((2, num_evts), np.nan, dtype = float)
        evt_wise_ant = np.full((2, num_ants, num_evts), np.nan, dtype = float)
        return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'bad_ant':bad_ant.astype(int),
            'snr_weights':snr_weights,
            'evt_wise':evt_wise,
            'evt_wise_ant':evt_wise_ant}
    else:
        del bad_path, bad_run_arr

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True)
    dt = wf_int.dt
    wf_len = wf_int.pad_len

    ara_mf = ara_matched_filter(st, run, dt, wf_len, get_sub_file = True)
    num_pols = ara_mf.num_pols
    del dt, wf_len, st, run

    evt_wise = np.full((num_pols, num_evts), np.nan, dtype = float)
    evt_wise_ant = np.full((num_pols, num_ants, num_evts), np.nan, dtype = float)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100:        
   
        if daq_qual_cut_sum[evt]:
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

        evt_wise[:, evt], evt_wise_ant[:, :, evt] = ara_mf.get_evt_wise_snr(wf_int.pad_v, wf_int.pad_num, snr_weights[:, evt]) 
        #print(evt_wise[:, evt])
        #print(np.nansum(evt_wise_ant[0, :8, evt]), np.nansum(evt_wise_ant[1, 8:, evt]))
        #print(evt_wise_ant[:, :, evt])
    del ara_root, num_evts, num_ants, wf_int, ara_mf, daq_qual_cut_sum, num_pols

    print('MF collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'bad_ant':bad_ant.astype(int),
            'snr_weights':snr_weights,
            'evt_wise':evt_wise,
            'evt_wise_ant':evt_wise_ant}







