import os
import h5py
import numpy as np
from tqdm import tqdm

def reco_sim_collector(Data, Station, Year):

    print('Collecting sim reco starts!')

    from tools.ara_sim_load import ara_root_loader
    from tools.ara_py_interferometers import py_interferometers
    from tools.ara_run_manager import get_path_info_v2
    from tools.ara_run_manager import get_example_run
    from tools.ara_constant import ara_const
    from tools.ara_wf_analyzer import wf_analyzer
    from tools.ara_py_interferometers import get_products
    from tools.ara_known_issue import known_issue_loader

    # const. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time 
   
    # config
    sim_type = get_path_info_v2(Data, 'AraOut.', '_')
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    flavor = int(get_path_info_v2(Data, 'AraOut.signal_F', '_A'))
    sim_run = int(get_path_info_v2(Data, 'txt.run', '.root'))
    if config < 6: year = 2015
    else: year = 2018
    if flavor != -1:
        s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_F{flavor}_A{Station}_R{config}.txt.run{sim_run}.h5'
    else:
        s_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_AraOut.{sim_type}_A{Station}_R{config}.txt.run{sim_run}.h5'
    print('snr_path:', s_path)
    snr_hf = h5py.File(s_path, 'r')
    snr = snr_hf['snr'][:]
    del s_path, snr_hf, sim_type, flavor, sim_run

    ex_run = get_example_run(Station, config)
    # bad antenna
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue, config

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, new_wf_time = wf_time)

    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, Station, year, run = ex_run, get_sub_file = True)
    wei_pairs = get_products(snr, ara_int.pairs, ara_int.v_pairs_len)
    del year, snr, ex_run

    # output array
    coef = np.full((2, 2, 2, num_evts), np.nan, dtype = float) # pol, rad, sol
    coord = np.full((2, 2, 2, 2, num_evts), np.nan, dtype = float) # pol, thephi, rad, sol

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True)
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, evt], sum_pol = True)
        coef[:, :, :, evt] = ara_int.coval
        coord[:, :, :, :, evt] = ara_int.coord
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])
        del wf_v
    del ara_root, num_evts, wf_int, ara_int, num_ants, wf_time, wei_pairs

    print('Reco snr mf collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'coef':coef,
            'coord':coord}

