import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.run import evt_num_loader
#from tools.chunk_samp_medi import vol_calib_collector_dat
from tools.chunk_samp_medi_temp import vol_calib_collector_dat

def wf_loader(CPath = curr_path, Data = None, Ped = None, Output = None, Act_Evt = None):

    # collecting info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    # event num    
    #evt_entry, add_tree = evt_num_loader(Station, Run, trig_set = 0, qual_set = 0, act_evt = Act_Evt)
    evt_entry = np.arange(10)
    add_tree = None

    save_wf = True
    # collecting wf
    medi_all, wf_all, wf_mean_all = vol_calib_collector_dat(Data, Ped, Station, Year, evt_entry, save_wf = save_wf)
    print(medi_all.shape)
    print(wf_all)
    print(wf_mean_all.shape)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name = f'Vol_Calib_A{Station}_R{Run}'
    if Act_Evt is not None:
        h5_file_name += f'_E{Act_Evt}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('evt_entry', data=evt_entry, compression="gzip", compression_opts=9)
    hf.create_dataset('medi_all', data=medi_all, compression="gzip", compression_opts=9)
    if save_wf == True:
        hf.create_dataset('wf_all', data=wf_all, compression="gzip", compression_opts=9)
        hf.create_dataset('wf_mean_all', data=wf_mean_all, compression="gzip", compression_opts=9)
    else:
        pass

    hf.close()
    del Station, Run, Config, Year, Month, Date
    del evt_entry, add_tree, medi_all, wf_all

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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Vol_Calib_Samp_Medi/>
    If you want more...
    <Event # ex)75030>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])
    act_evt = None
    if len(sys.argv) == 5:
        act_evt = int(sys.argv[4])

    wf_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output, Act_Evt = act_evt)

del curr_path












    
