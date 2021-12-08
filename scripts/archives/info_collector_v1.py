import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.run import config_collector
from tools.chunk import sub_info_collector_dat
from tools.wf import interpolation_bin_width

def info_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting info.
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)
    
    # collecting config.
    run_start_info, run_stop_info, masked_ant, rf_block_num, soft_block_num, trig_win_num, delay_enable, delay_num, month, date = config_collector(Data, Station, Run, Year)
    if Year == 2013 and np.isfinite(month) and np.isfinite(date):
        Month = np.copy(month)
        Date = np.copy(date)
        print('MMDD detected from RunStart file!',Month, Date)
    else:
        pass 

    # collecting sub-info.
    wf_len_all, wf_if_all, peak_all, rms_all, hill_all, evt_num, act_evt_num, ch_index, pol_type, trig_num, trig_chs, qual_num, qual_num_tot, unix_time, time_stamp, read_win, roll_mm, cliff_medi, freq_glitch = sub_info_collector_dat(Data, Ped, Station, Year)

    # dt info.
    dt_ns = interpolation_bin_width()

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Info_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')  
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)

    hf.create_dataset('run_start_info', data=run_start_info, compression="gzip", compression_opts=9)
    hf.create_dataset('run_stop_info', data=run_stop_info, compression="gzip", compression_opts=9)
    hf.create_dataset('masked_ant', data=masked_ant, compression="gzip", compression_opts=9) 
    hf.create_dataset('rf_block_num', data=rf_block_num, compression="gzip", compression_opts=9) 
    hf.create_dataset('soft_block_num', data=soft_block_num, compression="gzip", compression_opts=9) 
    hf.create_dataset('trig_win_num', data=trig_win_num, compression="gzip", compression_opts=9) 
    hf.create_dataset('delay_enable', data=delay_enable, compression="gzip", compression_opts=9) 
    hf.create_dataset('delay_num', data=delay_num, compression="gzip", compression_opts=9) 

    hf.create_dataset('dt_ns', data=np.array([dt_ns]), compression="gzip", compression_opts=9)
    hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_if_all', data=wf_if_all, compression="gzip", compression_opts=9)
    hf.create_dataset('peak_all', data=peak_all, compression="gzip", compression_opts=9)
    hf.create_dataset('rms_all', data=rms_all, compression="gzip", compression_opts=9)
    hf.create_dataset('hill_all', data=hill_all, compression="gzip", compression_opts=9)

    hf.create_dataset('evt_num', data=evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('act_evt_num', data=act_evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_ch_index', data=ch_index, compression="gzip", compression_opts=9)
    hf.create_dataset('pol_type', data=pol_type, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_num', data=trig_num, compression="gzip", compression_opts=9)
    hf.create_dataset('trig_chs', data=trig_chs, compression="gzip", compression_opts=9)
    hf.create_dataset('qual_num', data=qual_num, compression="gzip", compression_opts=9)
    hf.create_dataset('qual_num_tot', data=qual_num_tot, compression="gzip", compression_opts=9)
    hf.create_dataset('unix_time', data=unix_time, compression="gzip", compression_opts=9)
    hf.create_dataset('time_stamp', data=time_stamp, compression="gzip", compression_opts=9)
    hf.create_dataset('read_win', data=read_win, compression="gzip", compression_opts=9)
    hf.create_dataset('roll_mm', data=roll_mm, compression="gzip", compression_opts=9)
    hf.create_dataset('cliff_medi', data=cliff_medi, compression="gzip", compression_opts=9)
    hf.create_dataset('freq_glitch', data=freq_glitch, compression="gzip", compression_opts=9)
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













    
