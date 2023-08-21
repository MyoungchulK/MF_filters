import os
import numpy as np
from tqdm import tqdm
import h5py

def vertex_only_sim_collector(Data, Station, Year):

    print('Collecting vertex only sim starts!')

    from tools.ara_py_vertex import py_reco_handler
    from tools.ara_py_vertex import py_ara_vertex
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import get_file_name

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    num_pols_com = int(ara_const.POLARIZATION + 1)
    del ara_const

    # sub files
    h5_file_name = get_file_name(Data)
    ver_path = os.path.expandvars("$OUTPUT_PATH") + f'/ARA0{Station}/vertex_sim/vertex_{h5_file_name}.h5'
    print('vertex sim path:', ver_path)
    ver_hf = h5py.File(ver_path, 'r')
    entry_num = ver_hf['entry_num'][:]
    num_evts = len(entry_num)
    snr = ver_hf['snr'][:]
    hit = ver_hf['hit'][:]
    bad_ant = ver_hf['bad_ant'][:]
    del h5_file_name, ver_path, ver_hf

     # example run
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    ex_run = get_example_run(Station, config)
    del config

    # hit time
    hit_thres = 3
    if Station == 2: num_ants_cut = 3
    else: num_ants_cut = 2
    handler = py_reco_handler(Station, ex_run, 0.5, hit_thres, num_ants_cut = num_ants_cut, use_input_hit = True)
    del hit_thres, num_ants_cut

    # vertex
    vertex = py_ara_vertex(Station)
    del Station, ex_run

    # output array
    theta = np.full((num_pols_com, num_evts), np.nan, dtype = float)
    phi = np.copy(theta)

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        # get hit time
        handler.get_id_hits_prep_to_vertex(snr = snr[:, evt], hit = hit[:, evt])

        # get vertex reco
        #print(handler.pair_info, handler.useful_num_ants)
        vertex.get_pair_fit_spherical(handler.pair_info, handler.useful_num_ants)
        theta[:, evt] = vertex.theta
        phi[:, evt] = vertex.phi
        #print(theta[:, evt], phi[:, evt])
    del num_evts, num_ants, num_pols_com, handler, vertex

    print('Vertex only sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'snr':snr,
            'hit':hit,
            'theta':theta,
            'phi':phi}
