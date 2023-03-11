import numpy as np
import os, sys
import h5py
import click
from importlib import import_module
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker
from tools.ara_run_manager import get_path_info_v2

@click.command()
@click.option('-k', '--key', type = str)
@click.option('-s', '--station', type = int)
@click.option('-y', '--year', default = 2015, type = int)
@click.option('-d', '--data', type = str)
@click.option('-e', '--evt_range', default = [], multiple = True)
@click.option('-n', '--not_override', default = False, type = bool)
def script_loader(key, station, year, data, evt_range, not_override):

    if not_override:
        output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/{key}_sim/'
        slash_idx = data.rfind('/')
        dot_idx = data.rfind('.')
        data_name = data[slash_idx+1:dot_idx]
        if key == 'mf_noise' or key == 'mf_noise_debug':
            h5_file_name = f'{output}{key}_sim_A{station}_Evt{evt_range[0]}_{evt_range[1]}_{data_name}'
        else:
            h5_file_name = f'{output}{key}_{data_name}'
        h5_file_name_out = h5_file_name + '.h5'
        if os.path.exists(h5_file_name_out):
            print(f'{h5_file_name_out} is already there!!')
            return    

    sim_type = get_path_info_v2(data, 'AraOut.', '_')
    config = int(get_path_info_v2(data, '_R', '.txt'))
    flavor = int(get_path_info_v2(data, 'AraOut.signal_F', '_A'))
    sim_run = int(get_path_info_v2(data, 'txt.run', '.root'))
    if config < 6:
        year = 2015
    else:
        year = 2018
    print('St:', station, 'Type:', sim_type, 'Flavor:', flavor, 'Config:', config, 'Year:', year, 'Sim Run:', sim_run)
    del sim_type

    # run the chunk code
    module = import_module(f'tools.chunk_{key}_sim')
    method = getattr(module, f'{key}_sim_collector')
    if key == 'mf_noise' or key == 'mf_noise_debug':
        results = method(data, station, year, evt_range)
    else:
        results = method(data, station, year)
    del module, method

    # create output dir
    output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{station}/{key}_sim/'
    print(f'Output path check:{output}')
    if not os.path.exists(output):
        os.makedirs(output)

    slash_idx = data.rfind('/')
    dot_idx = data.rfind('.')
    data_name = data[slash_idx+1:dot_idx]
    del slash_idx, dot_idx
    if len(evt_range) != 0:
        evt_range = np.asarray(evt_range, dtype = int)
        print(evt_range)
    if key == 'mf_noise' or key == 'mf_noise_debug':
        h5_file_name = f'{output}{key}_sim_A{station}_Evt{evt_range[0]}_{evt_range[1]}_{data_name}'
    else:
        h5_file_name = f'{output}{key}_{data_name}'
    h5_file_name_out = h5_file_name + '.h5'
    hf = h5py.File(h5_file_name_out, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([station, sim_run, config, year, flavor]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name_out}')

    # quick size check
    size_checker(h5_file_name_out)

if __name__ == "__main__":

    script_loader()













    
