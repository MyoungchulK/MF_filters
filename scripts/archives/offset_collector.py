import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.qual import offset_block_checker

def offset_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting config.
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
        
    # collecting info.
    roll_max_mv, roll_max_t = offset_block_checker(Data, Ped, Station, Year = Year)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Offset_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')   
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('roll_max_mv', data=roll_max_mv, compression="gzip", compression_opts=9)
    hf.create_dataset('roll_max_t', data=roll_max_t, compression="gzip", compression_opts=9)
    hf.close()

    print(f'output is {Output}{h5_file_name}')
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) != 4:
        Usage = """
    This is designed to analyze all events in the data or simulation. Need to choose specific run or sim file.
    
    If it is data,
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Offset/>
    
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    offset_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
