import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.utility import size_checker
from tools.chunk_samp_map import samp_map_collector_dat

def samp_map_loader(Data = None, Ped = None, Output = None):

    # collecting info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
    #Station = 3
    #Run = 13030
    #Config = -1
    #Year = 2018
    #Month = 12
    #Date = 26

    # collecting wf
    results = samp_map_collector_dat(Data, Ped)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name = f'Samp_Map_A{Station}_R{Run}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    del Station, Run, Config, Year, Month, Date
    del results
    hf.close()

    print(f'output is {Output+h5_file_name}')

    # quick size check
    size_checker(Output+h5_file_name)
   
if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) < 4:
        Usage = """
    This is designed to analyze all events in the data or simulation. Need to choose specific run or sim file.

    If it is data,
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2015/unblinded/L1/ARA02/0104/run004783/event004783.root>
    <Pedestal file ex)/data/exp/ARA/2015/calibration/pedestals/ARA02/pedestalValues.run004781.dat>
    or <Pedestal file ex)/data/user/mkim/OMF_filter/ARA02/Ped/pedestalValues.run4783.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Samp_Map/>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    samp_map_loader(Data = data, Ped = ped, Output = output)

del curr_path












    
