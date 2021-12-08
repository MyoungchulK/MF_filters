import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
#from tools.chunk import wf_collector_dat
from tools.chunk import wf_collector_dat_v2
from tools.wf import time_pad_maker
from tools.wf import interpolation_bin_width

def wf_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    import psutil
    mem_u_tot = []
    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('start:',mem_mb)
    mem_u_tot.append(mem_mb)

    # collecting WF and info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
        
    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('data info:',mem_mb)
    mem_u_tot.append(mem_mb)

    #wf_all, wf_len_all, evt_num, trig_num, qual_num, final_chunk = wf_collector_dat(Data, Ped, Station, trig_set = 'all', qual_set = 'all', wf_len = True, evt_info = True)
    wf_all, wf_len_all, evt_num, trig_num, qual_num, final_chunk, mem_u = wf_collector_dat_v2(Data, Ped, Station, trig_set = 'all', qual_set = 'all', wf_len = True, evt_info = True)

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('after data extract:',mem_mb)
    mem_u_tot.append(mem_mb)

    time = time_pad_maker(p_dt = interpolation_bin_width())[0]

    mem_mb = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    print('after time pad:',mem_mb)
    mem_u_tot.append(mem_mb)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'WF_A{Station}_R{Run}_v2_cobalt.h5'
    hf = h5py.File(h5_file_name, 'w')   

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)

    hf.create_dataset('time', data=time, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_all', data=wf_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_num', data=trig_num, compression="gzip", compression_opts=9)
    hf.create_dataset('qual_num', data=qual_num, compression="gzip", compression_opts=9)
    hf.create_dataset('mem_u', data=mem_u, compression="gzip", compression_opts=9)
    hf.create_dataset('mem_u_tot', data=np.asarray(mem_u_tot), compression="gzip", compression_opts=9)

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    del Output, h5_file_name
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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Rayl/>
    
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    wf_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
