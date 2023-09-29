import os
import numpy as np
from tqdm import tqdm
import h5py

def qual_cut_3rd_bad_collector(st, run, qual_type = 3, analyze_blind_dat = False, no_tqdm = False):

    print('Quality cut 3rd bad starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_quality_cut import run_qual_cut_loader

    run_info = run_info_loader(st, run, analyze_blind_dat = analyze_blind_dat)
    q_dat = run_info.get_result_path(file_type = 'qual_cut_3rd', verbose = True)
    q_hf = h5py.File(q_dat, 'r+')
    tot_qual_cut = q_hf['tot_qual_cut'][:]
    del run_info, q_dat

    # run quality cut
    run_qual = run_qual_cut_loader(st, run, tot_qual_cut, analyze_blind_dat = analyze_blind_dat, qual_type = qual_type, verbose = True)
    bad_run = run_qual.get_bad_run_type()
    run_qual.get_bad_run_list()
    del run_qual, st, run, tot_qual_cut

    del q_hf['bad_run']
    q_hf.create_dataset('bad_run', data=bad_run, compression="gzip", compression_opts=9)
    q_hf.close()
    del bad_run 

    print('Quality cut 3rd bad is done!')

    return




