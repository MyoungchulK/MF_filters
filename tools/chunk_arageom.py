import os
import numpy as np
import h5py
from tqdm import tqdm

def arageom_collector(Data, Ped, analyze_blind_dat = False):

    print('ARA geom starts!')

    from tools.ara_constant import ara_const
    from tools.ara_data_load import ara_geom_loader
    from tools.ara_data_load import ara_uproot_loader

    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    ara_uproot = ara_uproot_loader(Data)
    st = ara_uproot.station_id
    run = ara_uproot.run
    yrs = ara_uproot.get_year(use_year = True)
    yrs_unix = ara_uproot.year
    del ara_uproot
    print('station id', st)
    print('run', run)
    print('year', yrs)
    print('year in unix time', yrs_unix)

    rf_ch = np.arange(num_ants, dtype = int)
    print('rf channel:', rf_ch)
    ara_geom = ara_geom_loader(st, yrs, verbose = True)
    ele_ch = ara_geom.get_ele_ch_idx()
    pol_ch = ara_geom.get_pol_ch_idx()
    trig_ch = ara_geom.get_trig_ch_idx()
    ant_xyz = ara_geom.get_ant_xyz()
    cable_delay = ara_geom.get_cable_delay()
    del ara_geom, num_ants

    print('ARA geom is done!')

    return {'st':np.asarray([st], dtype = int),
            'run':np.asarray([run], dtype = int),
            'yrs':np.asarray([yrs], dtype = int),
            'yrs_unix':np.asarray([yrs_unix], dtype = int),
            'rf_ch':rf_ch,
            'ele_ch':ele_ch,
            'pol_ch':pol_ch,
            'trig_ch':trig_ch,
            'ant_xyz':ant_xyz,
            'cable_delay':cable_delay}


