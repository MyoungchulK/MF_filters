import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.chunk_off_blk import raw_wf_collector_dat

def rms_peak_loader(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    # collecting wf
    blk_mean, rf_evt_num, ex_flag, freq_flag, freq_flag_sum, min_mu, min_cov_mtx, sig_decom, ratio, success_int, mu, wf_arr, time_pad, fft_arr, freq = raw_wf_collector_dat(Data, Ped, Station, Year)
    print(blk_mean.shape)
    print(rf_evt_num.shape)
    print(ex_flag.shape)
    print(freq_flag.shape)
    print(freq_flag_sum.shape)
    print(min_mu.shape)
    print(min_cov_mtx.shape)
    print(sig_decom.shape)
    print(ratio)
    print(bool(success_int))
    print(mu.shape)
    print(wf_arr.shape)
    print(time_pad.shape)
    print(fft_arr.shape)
    print(freq.shape)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name = f'Off_Blk_A{Station}_R{Run}'
    h5_file_name += f'.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('blk_mean', data=blk_mean, compression="gzip", compression_opts=9)
    hf.create_dataset('rf_evt_num', data=rf_evt_num, compression="gzip", compression_opts=9)
    hf.create_dataset('ex_flag', data=ex_flag, compression="gzip", compression_opts=9)
    hf.create_dataset('freq_flag', data=freq_flag, compression="gzip", compression_opts=9)
    hf.create_dataset('freq_flag_sum', data=freq_flag_sum, compression="gzip", compression_opts=9)
    hf.create_dataset('min_mu', data=min_mu, compression="gzip", compression_opts=9)
    hf.create_dataset('min_cov_mtx', data=min_cov_mtx, compression="gzip", compression_opts=9)
    hf.create_dataset('sig_decom', data=sig_decom, compression="gzip", compression_opts=9)
    hf.create_dataset('ratio', data=np.array([ratio]), compression="gzip", compression_opts=9)
    hf.create_dataset('success_int', data=np.array([success_int]), compression="gzip", compression_opts=9)
    hf.create_dataset('mu', data=mu, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_arr', data=wf_arr, compression="gzip", compression_opts=9)
    hf.create_dataset('time_pad', data=time_pad, compression="gzip", compression_opts=9)
    hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
    hf.create_dataset('fft_arr', data=fft_arr, compression="gzip", compression_opts=9)
    del Station, Run, Config, Year, Month, Date
    del blk_mean, rf_evt_num, ex_flag, freq_flag, freq_flag_sum, min_mu, min_cov_mtx, sig_decom, ratio, success_int, mu

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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Off_Blk/>

        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    rms_peak_loader(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path












    
