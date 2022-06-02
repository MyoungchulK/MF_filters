import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager_lite import run_info_loader
from tools.ara_utility import size_checker

def script_loader(Key = None, Station = None, Run = None, use_araroot_cut = False, use_mf_qual_cut = False, analyze_blind_dat = False):

    # get run info
    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)

    file_type = 'event'
    verbose = True    
    return_none = False
    return_dat_only = False
    Data, Ped = run_info.get_data_ped_path(file_type = file_type, return_none = return_none, verbose = verbose, return_dat_only = return_dat_only)
    Station, Run, Config, Year, Month, Date = run_info.get_data_info()
    del run_info   
 
    # run the chunk code
    module = import_module(f'tools.chunk_{Key}')
    method = getattr(module, f'{Key}_collector')
    results = method(Data, Ped, Station, Run, Year, use_araroot_cut = use_araroot_cut, use_mf_qual_cut = use_mf_qual_cut, analyze_blind_dat = analyze_blind_dat)
    del module, method

    # create output dir
    blind_type = ''
    if analyze_blind_dat:
        blind_type = '_full'
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}{blind_type}/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}{Key}{blind_type}_A{Station}_R{Run}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')
    
    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    del Key, Station, Run, Config, Year, Month, Date, Output
    del results
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)  
 
if __name__ == "__main__":

    if len (sys.argv) < 4:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Srtipt Key ex)rayl_lite>    
    <Station ex)3>
    <Run ex)11650>
    if you want quality cuts ...
    <AraRoot ex)0 (not use) or 1 (use)>
    <MF ex)0 (not use) or 1 (use)>
    if you want blinded data ...
    <blind_type ex)1>
  
        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    run=int(sys.argv[3])
    
    # bit messy...
    araroot=False
    mf=False
    blind_type = False
    if len(sys.argv) == 5:
        araroot = bool(int(sys.argv[4]))
    if len(sys.argv) == 6:
        araroot = bool(int(sys.argv[4]))
        mf = bool(int(sys.argv[5]))
    if len(sys.argv) == 7:
        araroot = bool(int(sys.argv[4]))
        mf = bool(int(sys.argv[5]))
        blind_type = bool(int(sys.argv[4]))

    script_loader(Key = key, Station = station, Run = run, use_araroot_cut = araroot, use_mf_qual_cut = mf, analyze_blind_dat = blind_type)













    
