import numpy as np
from tqdm import tqdm
import h5py

def reco_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_py_interferometers import get_products
    from tools.ara_run_manager import run_info_loader
    from tools.ara_known_issue import known_issue_loader

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
    yr = ara_uproot.year
    run = ara_uproot.run
    ara_root = ara_root_loader(Data, Ped, st, yr)
    del ara_uproot

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    daq_qual_cut_sum = daq_hf['daq_qual_cut_sum'][:]
    del daq_dat, daq_hf

    # snr info
    wei_key = 'snr'
    wei_dat = run_info.get_result_path(file_type = wei_key, verbose = True)
    wei_hf = h5py.File(wei_dat, 'r')
    if wei_key == 'mf':
        wei_ant = wei_hf['evt_wise_ant'][:]
        weights = np.full((num_ants, num_evts), np.nan, dtype = float)
        weights[:8] = wei_ant[0, :8]
        weights[8:] = wei_ant[1, 8:]
        del wei_ant 
    else:
        weights = wei_hf['snr'][:]
    del run_info, wei_key, wei_dat, wei_hf

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, add_double_pad = True)

    # interferometers
    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, st, yr, run = run, get_sub_file = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    wei_pairs = get_products(weights, pairs, v_pairs_len)
    del st, yr, run, pairs, v_pairs_len, weights

    # output array  
    coef = np.full((2, 2, 2, num_evts), np.nan, dtype = float) # pol, rad, sol
    coord = np.full((2, 2, 2, 2, num_evts), np.nan, dtype = float) # pol, thephi, rad, sol

    # loop over the events
    #for evt in tqdm(range(num_evts)):
    for evt in range(num_evts):
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

        coef[:, :, :, evt], coord[:, :, :, :, evt] = ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, evt])
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])       
    del ara_root, num_evts, num_ants, wf_int, ara_int, daq_qual_cut_sum, wei_pairs

    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'entry_num':entry_num,
            'trig_type':trig_type,
            'unix_time':unix_time,
            'pps_number':pps_number,
            'bad_ant':bad_ant,
            'coef':coef,
            'coord':coord}









