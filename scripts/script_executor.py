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
@click.option('-a', '--act_evt', default = None, multiple = True)
@click.option('-b', '--blind_dat', default = False, type = bool)
@click.option('-c', '--condor_run', default = False, type = bool)
def script_loader(key, station, run, act_evt, blind_dat, condor_run):

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
    elif key == 'blk_len' or key == 'evt_rate' or key == 'run_time' or key == 'ped' or key == 'qual_cut' or key == 'daq_cut' or key == 'ped_cut' or key == 'sub_info' or key == 'cw_time':
        return_dat_only = True
    Data, Ped = run_info.get_data_ped_path(file_type = file_type, return_none = return_none, verbose = verbose, return_dat_only = return_dat_only)
    station, run, Config, Year, Month, Date = run_info.get_data_info()
    del run_info   

    # move input to condor
    condor_info = condor_info_loader(use_condor = condor_run, verbose = True)
    Data = condor_info.get_target_to_condor_path(Data)
    Ped = condor_info.get_target_to_condor_path(Ped)
 
    # run the chunk code
    module = import_module(f'tools.chunk_{key}')
    method = getattr(module, f'{key}_collector')
    if key == 'wf':
        results = method(Data, Ped, analyze_blind_dat = blind_dat, sel_evts = act_evt)
    elif key == 'rayl_lite':
        results = method(Data, Ped, station, Year, analyze_blind_dat = blind_dat)
    elif key == 'l1':
        results = method(Data, Ped, station, run, Year, analyze_blind_dat = blind_dat)
    elif key == 'cw_time':
        results = method(station, run, analyze_blind_dat = blind_dat)
        return
    else:
        results = method(Data, Ped, analyze_blind_dat = blind_dat)
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
    if key == 'wf' and act_evt is not None:
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
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    del results
    hf.close()

    # move output from condor
    Output = condor_info.get_condor_to_target_path(h5_file_name, true_output_path)
    print(f'output is {Output}')

    # quick size check
    size_checker(Output)  
 
if __name__ == "__main__":

    script_loader()













    
