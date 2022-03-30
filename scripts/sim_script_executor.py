import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

def script_loader(Key = None, Station = None, Year = None, Data = None):

    # run the chunk code
    module = import_module(f'tools.chunk_{Key}_sim')
    method = getattr(module, f'{Key}_sim_collector')
    results = method(Data, Station, Year)
    del module, method

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}_sim/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    del slash_idx, dot_idx
    h5_file_name = f'{Output}{Key}_sim_A{Station}_{data_name}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')
    
    #saving result
    hf.create_dataset('config', data=np.array([Station]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    del results, Key, Station, Output
    hf.close()
    print(f'output is {h5_file_name}')

    # quick size check
    size_checker(h5_file_name)  
 
if __name__ == "__main__":

    if len (sys.argv) < 4:
        Usage = """

    Usage = python3 %s

    <Srtipt Key ex)mf>    
    <Station ex)2>
    <Year ex)2018>
    <Data ex)......> 

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    year=int(sys.argv[3])
    data=str(sys.argv[4])

    script_loader(Key = key, Station = station, Year = year, Data = data)













    
