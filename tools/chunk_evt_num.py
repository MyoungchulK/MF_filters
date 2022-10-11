import os
import numpy as np
import h5py
from tqdm import tqdm

def evt_num_collector(Data, Ped, analyze_blind_dat = False):

    print('Collecting event number starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_run_manager import run_info_loader

    # data config
    ara_uproot = ara_uproot_loader(Data)
    evt_num = ara_uproot.evt_num
    st = ara_uproot.station_id
    run = ara_uproot.run
    del ara_uproot

    # 100% data
    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    sub_dat = run_info.get_result_path(file_type = 'sub_info', verbose = True, force_blind = True)
    sub_hf = h5py.File(sub_dat, 'r')
    evt_num_full = sub_hf['evt_num'][:]
    del run_info, sub_dat, sub_hf

    # burn sample check
    evt_bool = np.in1d(evt_num, evt_num_full)
    bad_burn_flag = np.any(~evt_bool)
    if bad_burn_flag:
        print(f'Bad burn sample!!!!! A{st} R{run}')

    # align check
    if bad_burn_flag:
        align_bool = np.full((1), np.nan, dtype = float)
        evt_num_sub = np.copy(align_bool)
        align_diff = np.copy(align_bool)
        bad_align_flag = False
        bad_dupl_flag = False
    else:
        align_bool = np.in1d(evt_num_full, evt_num)
        evt_num_sub = evt_num_full[align_bool]
        if len(evt_num) != len(evt_num_sub):
            bad_dupl_flag = True
            print(f'Bad dupl sample!!!!! A{st} R{run}')
        else:
            bad_dupl_flag = False
        align_diff = (np.unique(evt_num).astype(int) - np.unique(evt_num_sub).astype(int)).astype(int)
        bad_align_flag = np.any(align_diff != 0)
        if bad_align_flag:
            print(f'Bad align sample!!!!! A{st} R{run}')
    bad_burn_flag = np.array([int(bad_burn_flag)], dtype = int)
    bad_align_flag = np.array([int(bad_align_flag)], dtype = int)
    bad_dupl_flag = np.array([int(bad_dupl_flag)], dtype = int)
    del st, run

    print('Event number collecting is done!')

    return {'evt_num':evt_num,
            'evt_num_full':evt_num_full,
            'evt_bool':evt_bool,
            'bad_burn_flag':bad_burn_flag,
            'align_bool':align_bool,
            'evt_num_sub':evt_num_sub,
            'align_diff':align_diff,
            'bad_align_flag':bad_align_flag,
            'bad_dupl_flag':bad_dupl_flag}







