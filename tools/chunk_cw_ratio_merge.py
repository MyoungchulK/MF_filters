import os
import numpy as np
import h5py
from tqdm import tqdm

def cw_ratio_merge_collector(Data, Station, Run, analyze_blind_dat = False, no_tqdm = False):

    print('Cw merge starts!')

    from tools.ara_run_manager import run_info_loader
    from tools.ara_utility import size_checker

    # load cw flag
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_ratio_old{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cw_dat = f'{output_path}cw_ratio{blind_type}_A{Station}_R{Run}.h5'
    print(f'cw_ratio_old_path:{cw_dat}', size_checker(f'{cw_dat}'))
    cw_hf = h5py.File(cw_dat, 'r')
    config = cw_hf['config'][:]
    evt_num = cw_hf['evt_num'][:]
    trig_type = cw_hf['trig_type'][:]
    bad_ant = cw_hf['bad_ant'][:]
    cw_ratio = cw_hf['cw_ratio'][:]
    del cw_dat, cw_hf

    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    cw_st_dat = run_info.get_result_path(file_type = 'cw_ratio_st', verbose = True, force_blind = True) # get the h5 file path
    cw_st_hf = h5py.File(cw_st_dat, 'r')
    cw_ratio_st = cw_st_hf['cw_ratio'][:]
    del cw_st_dat, cw_st_hf, run_info

    cw_ratio_merge = np.copy(cw_ratio)
    cw_nan = np.isnan(cw_ratio_st)
    cw_ratio_merge[~cw_nan] = cw_ratio_st[~cw_nan]

    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/cw_ratio{blind_type}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    h5_file_name = f'cw_ratio{blind_type}_A{Station}_R{Run}.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    hf.create_dataset('config', data=config, compression="gzip", compression_opts=9) 
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9) 
    hf.create_dataset('bad_ant', data=bad_ant, compression="gzip", compression_opts=9) 
    hf.create_dataset('trig_type', data=trig_type, compression="gzip", compression_opts=9) 
    hf.create_dataset('cw_ratio', data=cw_ratio_merge, compression="gzip", compression_opts=9) 
    hf.close()
    print(f'output is {output_path}{h5_file_name}.', size_checker(f'{output_path}{h5_file_name}'))
    
    print('cw merge is done!')

    return







