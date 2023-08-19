import numpy as np
from tqdm import tqdm

def vertex_collector(Data, Ped, analyze_blind_dat = False, use_l2 = False, no_tqdm = False):

    print('Collecting vertex starts!')

    if use_l2:
        from tools.ara_data_load import ara_l2_loader
    else:
        from tools.ara_data_load import ara_uproot_loader
        from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_vertex import py_reco_handler
    from tools.ara_py_vertex import py_ara_vertex
    from tools.ara_known_issue import known_issue_loader
    from tools.ara_quality_cut import get_bad_events

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_pols_com = int(ara_const.POLARIZATION + 1)
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

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, verbose = True, use_l2 = use_l2, analyze_blind_dat = analyze_blind_dat, st = st, run = run)

    # hit time 
    handler = py_reco_handler(st, run, wf_int.dt, 8)

    # vertex
    vertex = py_ara_vertex(st)
    del st, run

    # output array  
    snr = np.full((num_ants, num_evts), np.nan, dtype = float)
    hit = np.copy(snr)
    theta = np.full((num_pols_com, num_evts), np.nan, dtype = float)
    phi = np.copy(theta)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      #if evt == 500:        
        
        if daq_qual_cut_sum[evt]:
            continue

        # get entry and wf
        ara_root.get_entry(evt)
        ara_root.get_useful_evt(ara_root.cal_type.kLatestCalibWithOutTrimFirstBlock)
        
        # loop over the antennas
        for ant in range(num_ants):
            raw_t, raw_v = ara_root.get_rf_ch_wf(ant)
            wf_int.get_int_wf(raw_t, raw_v, ant, use_band_pass = True, use_cw = True, evt = evt)
            del raw_t, raw_v
            ara_root.del_TGraph()
        ara_root.del_usefulEvt()   

        # get hit time
        handler.get_id_hits_prep_to_vertex(wf_int.pad_v, wf_int.pad_t, wf_int.pad_num)
        snr[:, evt] = handler.snr_arr
        hit[:, evt] = handler.hit_time_arr
        #print(snr[:, evt], hit[:, evt])

        # get vertex reco
        #print(handler.pair_info, handler.useful_num_ants)
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        theta[:, evt] = vertex.theta
        phi[:, evt] = vertex.phi
        #print(theta[:, evt], phi[:, evt])
    del num_ants, num_pols_com, ara_root, num_evts, daq_qual_cut_sum, wf_int, handler, vertex 

    print('Vertex collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'snr':snr,
            'hit':hit,
            'theta':theta,
            'phi':phi}









