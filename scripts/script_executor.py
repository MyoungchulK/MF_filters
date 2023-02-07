import numpy as np
import os, sys
import h5py
import click
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader
from tools.ara_run_manager import condor_info_loader
from tools.ara_utility import size_checker

@click.command()
@click.option('-k', '--key', type = str)
@click.option('-s', '--station', type = int)
@click.option('-r', '--run', type = int)
@click.option('-a', '--act_evt', default = None)#, multiple = False)#, type = int)
@click.option('-b', '--blind_dat', default = False, type = bool)
@click.option('-c', '--condor_run', default = False, type = bool)
@click.option('-n', '--not_override', default = False, type = bool)
@click.option('-l', '--l2_data', default = False, type = bool)
@click.option('-q', '--qual_2nd', default = False, type = bool)
@click.option('-t', '--no_tqdm', default = False, type = bool)
def script_loader(key, station, run, act_evt, blind_dat, condor_run, not_override, l2_data, qual_2nd, no_tqdm):

    if not_override:
        blind_type = ''
        if blind_dat:
            blind_type = '_full'
        true_output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/{key}{blind_type}/'
        file_format = '.h5'
        sub_key = ''
        if key == 'ped':
            file_format = '.dat'
            sub_key = '_values'
        h5_file_name = f'{key}{blind_type}{sub_key}_A{station}_R{run}'
        done_path = f'{true_output_path}{h5_file_name}{file_format}'
        if os.path.exists(done_path):
            try:
                hf = h5py.File(f'{done_path}', 'r')
                if len(list(hf)) == 0:
                    print(f'{done_path} is empty!!')
                else:
                    print(f'{done_path} is already there!!')
                    return
            except OSError:
                print(f'{done_path} is corrupted!!')

    # get run info
    run_info = run_info_loader(station, run, analyze_blind_dat = blind_dat)

    file_type = 'event'
    verbose = True    
    return_none = False
    return_dat_only = False
    if key == 'sensor':
        file_type = 'sensorHk'
        return_none = True
        return_dat_only = True
    elif key == 'l1':
        file_type = 'eventHk'
        return_none = True
        return_dat_only = True
    elif key == 'blk_len' or key == 'rf_len' or key == 'dead' or key == 'dupl' or  key == 'run_time' or key == 'ped' or key == 'cw_band' or key == 'qual_cut' or key == 'evt_num' or key == 'medi' or key == 'sub_info' or key == 'cw_time':
        return_dat_only = True
    Data, Ped = run_info.get_data_ped_path(file_type = file_type, return_none = return_none, verbose = verbose, return_dat_only = return_dat_only, l2_data = l2_data)
    station, run, Config, Year, Month, Date = run_info.get_data_info()
    del run_info   

    # move input to condor
    condor_info = condor_info_loader(use_condor = condor_run, verbose = True)
    Data = condor_info.get_target_to_condor_path(Data)
    Ped = condor_info.get_target_to_condor_path(Ped)
 
    # run the chunk code
    module = import_module(f'tools.chunk_{key}')
    method = getattr(module, f'{key}_collector')
    if key == 'wf' or key == 'wf_ver2':
        results = method(Data, Ped, analyze_blind_dat = blind_dat, sel_evts = act_evt)
    elif key == 'qual_cut':
        results = method(Data, Ped, analyze_blind_dat = blind_dat, qual_2nd = qual_2nd, no_tqdm = no_tqdm)
    elif key == 'l1':
        results = method(Data, Ped, station, run, Year, analyze_blind_dat = blind_dat)
    elif key == 'ped':
        results = method(Data, station, run, analyze_blind_dat = blind_dat)
        return
    elif key == 'cw_band':
        results = method(Data, station, run, analyze_blind_dat = blind_dat, no_tqdm = no_tqdm)
    elif key == 'cw_time':
        results = method(station, run, analyze_blind_dat = blind_dat)
        return
    elif key == 'l2' or key == 'l2_temp':
        results = method(Data, Ped, analyze_blind_dat = blind_dat, use_condor = condor_run)
        return 
    else:
        results = method(Data, Ped, analyze_blind_dat = blind_dat, use_l2 = l2_data, no_tqdm = no_tqdm)
    del module, method

    # create output dir
    blind_type = ''
    if blind_dat:
        blind_type = '_full'
    true_output_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/{key}{blind_type}/'
    if not os.path.exists(true_output_path):
        os.makedirs(true_output_path)
    if condor_run:
        output_path = condor_info.local_path
    else:
        output_path = true_output_path
    print(f'Output path check:{output_path}')
    h5_file_name = f'{key}{blind_type}_A{station}_R{run}'
    if (key == 'wf' or key == 'wf_ver2') and act_evt is not None:
        act_evt = act_evt.split(',')
        act_evt = np.asarray(act_evt).astype(int)
        if len(act_evt) == 1:
            h5_file_name += f'_E{act_evt[0]}'
        else:
            h5_file_name += f'_E{act_evt[0]}_to_E{act_evt[-1]}'
    h5_file_name += f'.h5'
    hf = h5py.File(f'{output_path}{h5_file_name}', 'w')
    
    #saving result
    hf.create_dataset('config', data=np.array([station, run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        try:
            hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
        except TypeError:
            dt = h5py.vlen_dtype(np.dtype(float))
            hf.create_dataset(r, data=results[r], dtype = dt, compression="gzip", compression_opts=9)
    del results
    hf.close()

    # move output from condor
    Output = condor_info.get_condor_to_target_path(h5_file_name, true_output_path)
    print(f'output is {Output}.', size_checker(Output))

if __name__ == "__main__":

    script_loader()













    
