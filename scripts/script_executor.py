import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader
from tools.utility import size_checker

def script_loader(Key = None, Station = None, Run = None):

    # get run info
    run_info = run_info_loader(Station, Run, analyze_blind_dat = False)
    #Data, Ped = run_info.get_data_ped_path(file_type = 'sensorHk', verbose = True)
    Data, Ped = run_info.get_data_ped_path(verbose = True, return_dat_only = False)
    Station, Run, Config, Year, Month, Date = run_info.get_data_info()
    del run_info   
 
    # run the chunk code
    module = import_module(f'tools.chunk_{Key}')
    method = getattr(module, f'{Key}_collector')
    results = method(Data, Ped)
    del module, method

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}{Key}_A{Station}_R{Run}.h5'
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

    <Srtipt Key ex)qual_cut>    
    <Station ex)2>
    <Run ex)11650>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    run=int(sys.argv[3])

    script_loader(Key = key, Station = station, Run = run)













    
