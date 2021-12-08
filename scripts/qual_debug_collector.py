import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.qual_debug import median_tilt_maker

def info_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting info.
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
    
    # collecting debug
    medi_all, wf_if_all, wf_len_all = median_tilt_maker(Data, Ped, Station, Year)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Medi_Tilt_repeder_A{Station}_R{Run}.h5'
    #h5_file_name=f'Medi_Tilt_kJustPed_A{Station}_R{Run}.h5'
    #h5_file_name=f'Medi_Tilt_kLatestCalib_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')  
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('medi_all', data=medi_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_if_all', data=wf_if_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    file_size = np.round(os.path.getsize(Output+h5_file_name)/1204/1204,2)
    print('file size is', file_size, 'MB')
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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Medi_Tilt/>
    
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    info_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
