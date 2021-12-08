import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.chunk import wf_collector_dat
from tools.wf import interpolation_bin_width

def info_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting config.
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
        
    # collecting info.
    wf_all, wf_len_all, wf_if_all, peak_all, rms_all, hill_all, raw_wf_len_all, raw_wf_if_all, raw_peak_all, raw_rms_all, raw_hill_all, evt_num, act_evt_num, ch_index, pol_type, trig_num, trig_chs, qual_num, unix_time, time_stamp, read_win, final_chunk = wf_collector_dat(Data, Ped, Station,
                            wf_len = True, wf_if_info = True, peak_dat = True, rms_dat = True, hill_dat = True, 
                            raw_wf_len = True, raw_wf_if_info = True, raw_peak_dat = True, raw_rms_dat = True, raw_hill_dat = True,
                            evt_info = True, act_evt_info = True, trig_info = True, pol_info = True, unix_info = True, time_stamp_info = True, read_win_info = True, Year = Year)

    # dt info.
    dt_ns = interpolation_bin_width()

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Info_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')   
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('dt_ns', data=np.array([dt_ns]), compression="gzip", compression_opts=9)
    hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_if_all', data=wf_if_all, compression="gzip", compression_opts=9)
    hf.create_dataset('peak_all', data=peak_all, compression="gzip", compression_opts=9)
    hf.create_dataset('rms_all', data=rms_all, compression="gzip", compression_opts=9)
    hf.create_dataset('hill_all', data=hill_all, compression="gzip", compression_opts=9)
    hf.create_dataset('raw_wf_len_all', data=raw_wf_len_all, compression="gzip", compression_opts=9)
    hf.create_dataset('raw_wf_if_all', data=raw_wf_if_all, compression="gzip", compression_opts=9)
    hf.create_dataset('raw_peak_all', data=raw_peak_all, compression="gzip", compression_opts=9)
    hf.create_dataset('raw_rms_all', data=raw_rms_all, compression="gzip", compression_opts=9)
    hf.create_dataset('raw_hill_all', data=raw_hill_all, compression="gzip", compression_opts=9)
    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('act_evt_num', data=act_evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('ch_index', data=ch_index, compression="gzip", compression_opts=9)
    hf.create_dataset('pol_type', data=pol_type, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_num', data=trig_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_chs', data=trig_chs, compression="gzip", compression_opts=9)
    hf.create_dataset('qual_num', data=qual_num, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('time_stamp', data=time_stamp, compression="gzip", compression_opts=9)
    hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)
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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Info/>
    
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













    
