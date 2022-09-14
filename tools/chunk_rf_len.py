import numpy as np
import h5py

def rf_len_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting rayl. starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_constant import ara_const
    from tools.ara_run_manager import run_info_loader

    # geom. info.
    ara_const = ara_const()
    num_ddas = ara_const.DDA_PER_ATRI
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    trig_type = ara_uproot.get_trig_type()
    st = ara_uproot.station_id
    run = ara_uproot.run
    blk_len = (ara_uproot.read_win // num_ddas).astype(float)

    # pre quality cut
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    daq_dat = run_info.get_result_path(file_type = 'qual_cut', verbose = True)
    daq_hf = h5py.File(daq_dat, 'r')
    tot_qual_cut_sum = daq_hf['tot_qual_cut_sum'][:]
    cw_dat = run_info.get_result_path(file_type = 'cw_cut', verbose = True)
    cw_hf = h5py.File(cw_dat, 'r')
    cw_qual_cut_sum = cw_hf['cw_qual_cut_sum'][:]
    del daq_dat, daq_hf, cw_dat, cw_hf, run_info
   
    # clean soft trigger 
    tot_cuts = (tot_qual_cut_sum + cw_qual_cut_sum).astype(int)
    clean_rf_idx = np.logical_and(tot_cuts == 0, trig_type == 0)

    # output
    rf_len = (blk_len * 20 / 0.5).astype(int)
    rf_len = rf_len[clean_rf_idx]
    print(rf_len)

    print('Rayl. collecting is done!')

    return {'rf_len':rf_len}



