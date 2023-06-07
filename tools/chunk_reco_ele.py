import numpy as np
from tqdm import tqdm
import h5py

def reco_ele_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting reco ele. starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_py_interferometers import get_products
    from tools.ara_run_manager import run_info_loader
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

    # pre quality cut
    if use_l2 == False:
        daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, qual_type = 2)[0]

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # snr info
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
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
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # interferometers
    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, st, yr, run = run, get_sub_file = True, use_ele_max = True, verbose = True)
    pairs = ara_int.pairs
    v_pairs_len = ara_int.v_pairs_len
    num_rads = ara_int.num_rads
    num_thetas = ara_int.num_thetas
    num_ray_sol = ara_int.num_ray_sol
    wei_pairs = get_products(weights, pairs, v_pairs_len)
    del st, yr, run, pairs, v_pairs_len, weights

    # output array  
    coef = np.full((num_pols, num_rads, num_ray_sol, num_evts), np.nan, dtype = float) # pol, rad, sol
    coef_ele = np.full((num_pols, num_thetas, num_rads, num_ray_sol, num_evts), np.nan, dtype = float) # pol, theta, rad, sol
    coord = np.full((num_pols, 2, num_rads, num_ray_sol, num_evts), np.nan, dtype = float) # pol, thephi, rad, sol
    del num_pols, num_rads, num_ray_sol, num_thetas

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
            wf_int.get_int_wf(raw_t, raw_v, ant, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, evt])
        coef[:, :, :, evt] = ara_int.coval_max
        coef_ele[:, :, :, :, evt] = ara_int.coval_ele_max
        coord[:, :, :, :, evt] = ara_int.coord_max
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])       
    del ara_root, num_evts, num_ants, wf_int, ara_int, daq_qual_cut_sum, wei_pairs

    print('Reco ele. collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'coef':coef,
            'coef_ele':coef_ele,
            'coord':coord}









