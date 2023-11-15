import numpy as np
from tqdm import tqdm
import h5py

def reco_ele_lite_q2_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting reco ele lite quator starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers_ele import py_interferometers
    from tools.ara_py_interferometers_ele import get_products
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
        del ara_uproot, yr

    # pre quality cut
    if use_l2 == False:
        daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, qual_type = 2)[0]

    known_issue = known_issue_loader(st)
    bad_ant = known_issue.get_bad_antenna(run, print_integer = True)
    del known_issue

    # sub info
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    wei_dat = run_info.get_result_path(file_type = 'snr', verbose = True)
    wei_hf = h5py.File(wei_dat, 'r')
    weights = wei_hf['snr'][:]
    evt_num_b = np.full((10), -1, dtype = int)
    if analyze_blind_dat:
        print('BURN!!!!!')
        reco_dat = run_info.get_result_path(file_type = 'reco_ele_lite', verbose = True, return_none = True, force_unblind = True)
        if reco_dat is not None:
            reco_hf = h5py.File(reco_dat, 'r')
            evt_num_b = reco_hf['evt_num'][:]
            coef_cal_b = reco_hf['coef_cal'][:]
            coord_cal_b = reco_hf['coord_cal'][:]
            coef_max_b = reco_hf['coef_max'][:]
            coord_max_b = reco_hf['coord_max'][:]
            coef_s_max_b = reco_hf['coef_s_max'][:]
            coord_s_max_b = reco_hf['coord_s_max'][:]
            del reco_hf
    del wei_dat, wei_hf, run_info

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # interferometers
    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, st, run = run, get_sub_file = True, verbose = True, use_only_max = analyze_blind_dat)
    only_max = ara_int.use_only_max
    if only_max:
        num_angs = ara_int.num_angs
    else:
        radius = ara_int.radius
        theta = ara_int.theta
        re_shape = ara_int.results_shape
    wei_pairs = get_products(weights, ara_int.pairs, ara_int.v_pairs_len)
    del st, run, weights

    # output array 
    if only_max: 
        coef_cal = np.full((num_pols, num_evts), np.nan, dtype = float) # pol, evt
        coord_cal = np.full((num_angs + 1, num_pols, num_evts), np.nan, dtype = float) # thepiz, pol, evt 
        coef_max = np.copy(coef_cal) # pol, evt
        coord_max = np.full((num_angs + 2, num_pols, num_evts), np.nan, dtype = float) # thepirz, pol, evt
        coef_s_max = np.copy(coef_cal) # pol, evt
        coord_s_max = np.copy(coord_max) # thepirz, pol, evt
        del num_angs
        if reco_dat is not None:
            print('Filling burn sample results into blind data!')
            for e in tqdm(range(len(evt_num_b)), disable = no_tqdm):
                evt_idx = np.where(evt_num == evt_num_b[e])[0]
                if len(evt_idx) > 0:
                    evt_d = evt_idx[0]
                    coef_cal[:, evt_d] = coef_cal_b[:, e]
                    coord_cal[:, :, evt_d] = coord_cal_b[:, :, e]
                    coef_max[:, evt_d] = coef_max_b[:, e]
                    coord_max[:, :, evt_d] = coord_max_b[:, :, e]
                    coef_s_max[:, evt_d] = coef_s_max_b[:, e]
                    coord_s_max[:, :, evt_d] = coord_s_max_b[:, :, e]
                    del evt_d
            del coef_cal_b, coord_cal_b, coef_max_b, coord_max_b, coef_s_max_b, coord_s_max_b
        del reco_dat
    else:
        coef = np.full((re_shape[0], re_shape[1], re_shape[2], re_shape[3], num_evts), np.nan, dtype = float) # pol, theta, rad, ray, evt
        coord = np.copy(coef) # pol, theta, rad, ray, evt
        del re_shape
    del num_pols

    half_num_evts = num_evts // 2
    q_num_evts = num_evts // 4 - 1
    sel_range = np.arange(num_evts, dtype = int)
    sel_evt_idx = np.logical_and(sel_range > q_num_evts, sel_range < half_num_evts).astype(int)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 0:        
      if evt > q_num_evts and evt < half_num_evts:       
 
        if daq_qual_cut_sum[evt]:
            continue
        if analyze_blind_dat == True and evt_num[evt] in evt_num_b: 
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
        if only_max:
            coef_cal[:, evt] = ara_int.coef_cal
            coord_cal[:, :, evt] = ara_int.coord_cal
            coef_max[:, evt] = ara_int.coef_max
            coord_max[:, :, evt] = ara_int.coord_max
            coef_s_max[:, evt] = ara_int.coef_s_max
            coord_s_max[:, :, evt] = ara_int.coord_s_max
        else:
            coef[:, :, :, :, evt] = ara_int.coef_max_ele
            coord[:, :, :, :, evt] = ara_int.coord_max_ele
    del ara_root, num_evts, num_ants, wf_int, ara_int, daq_qual_cut_sum, wei_pairs, evt_num_b

    print('Reco ele lite quator collecting is done!')

    if only_max:
        return {'evt_num':evt_num,
                'sel_evt_idx':sel_evt_idx,
                'trig_type':trig_type,
                'bad_ant':bad_ant,
                'coef_cal':coef_cal,
                'coord_cal':coord_cal,
                'coef_max':coef_max,
                'coord_max':coord_max,
                'coef_s_max':coef_s_max,
                'coord_s_max':coord_s_max}
    else:
        return {'evt_num':evt_num,
                'sel_evt_idx':sel_evt_idx,
                'trig_type':trig_type,
                'bad_ant':bad_ant,
                'theta':theta,
                'radius':radius,
                'coef':coef,
                'coord':coord}









