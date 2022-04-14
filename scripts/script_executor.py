import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader
from tools.ara_utility import size_checker

def script_loader(Key = None, Station = None, Run = None, Act_Evt = None, analyze_blind_dat = False):

    # get run info
    run_info = run_info_loader(Station, Run, analyze_blind_dat = analyze_blind_dat)
    
    if Key == 'sensor':
        Data, Ped = run_info.get_data_ped_path(file_type = 'sensorHk', return_none = True, verbose = True, return_dat_only = True)
    elif Key == 'l1':
        Data, Ped = run_info.get_data_ped_path(file_type = 'eventHk', return_none = True, verbose = True, return_dat_only = True)
    elif Key == 'blk_len' or Key == 'evt_rate' or Key == 'run_time':
        Data, Ped = run_info.get_data_ped_path(verbose = True, return_dat_only = True)
    else:
        Data, Ped = run_info.get_data_ped_path(verbose = True, return_dat_only = False)
    Station, Run, Config, Year, Month, Date = run_info.get_data_info()
    del run_info   
 
    # run the chunk code
    module = import_module(f'tools.chunk_{Key}')
    method = getattr(module, f'{Key}_collector')
    if Key == 'wf':
        results = method(Data, Ped, analyze_blind_dat = analyze_blind_dat, sel_evts = Act_Evt)
    elif Key == 'l1':
        results = method(Data, Ped, Station, Run, Year, analyze_blind_dat = analyze_blind_dat)
    else:
        results = method(Data, Ped, analyze_blind_dat = analyze_blind_dat)
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
    if Key == 'wf' and Act_Evt is not None:
        if len(Act_Evt) == 1:
            h5_file_name += f'_E{Act_Evt[0]}'
        else:
            h5_file_name += f'_E{Act_Evt[0]}_to_E{Act_Evt[-1]}'
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

    <Srtipt Key ex)qual_cut>    
    <Station ex)2>
    <Run ex)11650>
    if Key is wf and want more...
    <Event # ex)75030>
    if you want blinded data ...
    <blind_type ex)1>
    

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    run=int(sys.argv[3])
    act_evt = None
    blind_type = False
    if len(sys.argv) == 5:
        blind_type = bool(int(sys.argv[4]))
    act_evt = None
    if len(sys.argv) > 5:
        blind_type = bool(int(sys.argv[4]))
        act_evt = np.asarray(sys.argv[5].split(','), dtype = int)

    script_loader(Key = key, Station = station, Run = run, Act_Evt = act_evt, analyze_blind_dat = blind_type)













    
