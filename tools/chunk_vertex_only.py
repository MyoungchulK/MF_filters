import numpy as np
from tqdm import tqdm
import h5py

def vertex_only_collector(st, run, analyze_blind_dat = False, no_tqdm = False):

    print('Collecting vertex starts!')

    from tools.ara_constant import ara_const
    from tools.ara_py_vertex import py_reco_handler
    from tools.ara_py_vertex import py_ara_vertex
    from tools.ara_quality_cut import get_bad_events
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_pols_com = int(ara_const.POLARIZATION + 1)
    del ara_const

    # snr info
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    ver_dat = run_info.get_result_path(file_type = 'vertex', verbose = True)
    ver_hf = h5py.File(ver_dat, 'r')
    evt_num = ver_hf['evt_num'][:]
    num_evts = len(evt_num)
    trig_type = ver_hf['trig_type'][:]
    snr = ver_hf['snr'][:]
    hit = ver_hf['hit'][:]
    bad_ant = ver_hf['bad_ant'][:]
    del run_info, ver_dat, ver_hf

    # pre quality cut
    daq_qual_cut_sum = get_bad_events(st, run, analyze_blind_dat = analyze_blind_dat, verbose = True, evt_num = evt_num, qual_type = 2)[0]

    # hit time
    hit_thres = 3
    if st == 2: num_ants_cut = 3
    else: num_ants_cut = 2 
    handler = py_reco_handler(st, run, 0.5, hit_thres, num_ants_cut = num_ants_cut, use_input_hit = True)
    del hit_thres, num_ants_cut

    # vertex
    vertex = py_ara_vertex(st)
    del st, run

    # output array  
    theta = np.full((num_pols_com, num_evts), np.nan, dtype = float)
    phi = np.copy(theta)

    # loop over the events
    for evt in tqdm(range(num_evts), disable = no_tqdm):
      if evt == 500:        
        
        if daq_qual_cut_sum[evt]:
            continue

        # get hit time
        handler.get_id_hits_prep_to_vertex(snr = snr[:, evt], hit = hit[:, evt])

        # get vertex reco
        #print(handler.pair_info, handler.useful_num_ants)
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        theta[:, evt] = vertex.theta
        phi[:, evt] = vertex.phi
        #print(handler.snr_arr, handler.hit_time_arr)
        #print(theta[:, evt], phi[:, evt])
    del num_ants, num_pols_com, num_evts, daq_qual_cut_sum, handler, vertex 

    print('Vertex only collecting is done!')

    return {'evt_num':evt_num,
            'trig_type':trig_type,
            'bad_ant':bad_ant,
            'snr':snr,
            'hit':hit,
            'theta':theta,
            'phi':phi}









