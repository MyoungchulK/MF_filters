import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.run import evt_num_loader
from tools.chunk_raw_random import raw_wf_collector_dat

def raw_wf_loader(CPath = curr_path, Data = None, Ped = None, Output = None, Ran_Evt = False):

    # collecting info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    # event num 
    if Ran_Evt == True:
        evt_entry_input = None
    else:  
        #path = f'/data/user/mkim/OMF_filter/ARA0{Station}/Random_WF_New/'
        path = f'/home/mkim/test/'
        h5_name = f'Random_WF_A{Station}_R{Run}.h5' 
        hf = h5py.File(path+h5_name, 'r') 
        evt_entry_input = hf['evt_entry'][:]
        print(f'{path+h5_name} is loaded!')
        print('Inputted evt entry:', evt_entry_input)
        del hf

    # collecting wf
    wf_all, evt_entry = raw_wf_collector_dat(Data, Ped, Station, Year, evt_entry_input)
    print('Outputted evt entry:',evt_entry) 

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name = f'Random_WF_A{Station}_R{Run}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('evt_entry', data=evt_entry, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_all', data=wf_all, compression="gzip", compression_opts=9)
    del Station, Run, Config, Year, Month, Date
    del evt_entry, wf_all, evt_entry_input 

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    file_size = np.round(os.path.getsize(Output+h5_file_name)/1204/1204,2)
    print('file size is', file_size, 'MB')
    del Output, h5_file_name, file_size
    print('done!')
   
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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Random_WF_New/>
    If you want more...
    <Ran_evt ex) True>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])
    #ran_evt = True
    ran_evt = False
    if len(sys.argv) == 5:
        ran_evt = str(sys.argv[4])

    raw_wf_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output,Ran_Evt = ran_evt)

del curr_path












    
