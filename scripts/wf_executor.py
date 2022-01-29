import numpy as np
import os, sys
import h5py
from importlib import import_module

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import run_info_loader
from tools.utility import size_checker

def script_loader(Key = None, Station = None, Run = None, Act_Evt = None):

    # get run info
    #run_info = run_info_loader(Station, Run, analyze_blind_dat = False)
    #Data, Ped = run_info.get_data_ped_path(verbose = True, return_dat_only = True)
    #Station, Run, Config, Year, Month, Date = run_info.get_data_info()
    #del run_info   

    Data = '/data/exp/ARA/2022/filtered/L0/ARA01/0101/run23291/event23291.root'
    #Ped = '/home/mkim/ped/run_023290/pedestalValues.run023290.dat'
    Ped = '0'
    Station = 1
    Run = 23291
    Config = -1
    Year = 2022
    Month = 1
    Date = 1 

    # run the chunk code
    module = import_module(f'tools.chunk_{Key}')
    method = getattr(module, f'{Key}_collector')
    results = method(Data, Ped, Station, Year, sel_evts = Act_Evt)
    del module, method

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)
    h5_file_name = f'{Output}{Key}_A{Station}_R{Run}'
    if Act_Evt is not None:
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

    <Srtipt Key ex)wf>    
    <Station ex)2>
    <Run ex)11650>
    If you want more...
    <Event # ex)75030>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    run=int(sys.argv[3])
    act_evt = None
    if len(sys.argv) == 5:
        act_evt = np.asarray(sys.argv[4].split(','), dtype = int)

    script_loader(Key = key, Station = station, Run = run, Act_Evt = act_evt)













    
