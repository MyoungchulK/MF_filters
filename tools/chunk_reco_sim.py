import os
import h5py
import numpy as np
from tqdm import tqdm

def reco_sim_collector(Data, Station, Year):

    print('Collecting reco sim starts!')

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
    num_pols = ara_const.POLARIZATION
    del ara_const

    # data config
    ara_root = ara_root_loader(Data, Station, Year)
    ara_root.get_sub_info(Data, get_angle_info = False)
    num_evts = ara_root.num_evts
    entry_num = ara_root.entry_num
    wf_time = ara_root.wf_time 
 
    # bad antenna
    config = int(get_path_info_v2(Data, '_R', '.txt'))
    ex_run = get_example_run(Station, config)
    known_issue = known_issue_loader(Station)
    bad_ant = known_issue.get_bad_antenna(ex_run, print_integer = True)
    del known_issue, config
 
    # sub files
    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    h5_file_name = Data[slash_idx+1:dot_idx]
    band_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_band_sim/cw_band_{h5_file_name}.h5'
    snr_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/snr_sim/snr_{h5_file_name}.h5'
    print('cw band sim path:', band_path)
    print('snr sim path:', snr_path)
    del slash_idx, dot_idx, h5_file_name 

    # wf analyzer
    wf_int = wf_analyzer(use_time_pad = True, use_band_pass = True, use_cw = True, new_wf_time = wf_time, sim_path = band_path)
    del band_path

    ara_int = py_interferometers(wf_int.pad_len, wf_int.dt, Station, Year, run = ex_run, get_sub_file = True)
    num_rads = ara_int.num_rads
    num_ray_sol = ara_int.num_ray_sol
    snr_hf = h5py.File(snr_path, 'r')
    snr = snr_hf['snr'][:]
    wei_pairs = get_products(snr, ara_int.pairs, ara_int.v_pairs_len)
    del Year, ex_run, snr_path, snr_hf, snr

    # output array
    coef = np.full((num_pols, num_rads, num_ray_sol, num_evts), np.nan, dtype = float) # pol, rad, sol
    coord = np.full((num_pols, 2, num_rads, num_ray_sol, num_evts), np.nan, dtype = float) # pol, thephi, rad, sol
    del num_pols, num_rads, num_ray_sol

    # loop over the events
    for evt in tqdm(range(num_evts)):
      #if evt <100: # debug 

        wf_v = ara_root.get_rf_wfs(evt)
        for ant in range(num_ants):
            wf_int.get_int_wf(wf_time, wf_v[:, ant], ant, use_sim = True, use_zero_pad = True, use_band_pass = True, use_cw = True, evt = evt)
        ara_int.get_sky_map(wf_int.pad_v, weights = wei_pairs[:, evt])
        coef[:, :, :, evt] = ara_int.coval_max
        coord[:, :, :, :, evt] = ara_int.coord_max
        #print(coef[:, :, :, evt], coord[:, :, :, :, evt])
        del wf_v
    del ara_root, num_evts, wf_int, ara_int, num_ants, wf_time, wei_pairs

    print('Reco sim collecting is done!')

    return {'entry_num':entry_num,
            'bad_ant':bad_ant,
            'coef':coef,
            'coord':coord}

