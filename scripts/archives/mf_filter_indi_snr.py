import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.chunk import wf_collector_dat
from tools.chunk import wf_collector_sim
from tools.fft import freq_pad_maker
from tools.fft import fft_maker
from tools.sc import ntot_loader
from tools.sc import sc_loader
from tools.temp import temp_loader
from tools.mf import band_square
from tools.mf import mf_max_snr_lag

def matched_filter_indi_snr(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting WF and info
    if Ped is not None: # data
        Station, Run, Config, Year, Month, Date = data_info_reader(Data)
        dat_all = wf_collector_dat(Data, Ped, Station, trig_set = 2, qual_set = 1)[0]
    else: # sim.
        Station = 2 
        Run = -1
        Config = -1
        Year = -1
        Month = -1 
        Date = -1
        dat_all = wf_collector_sim(Data)[0]

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    if Ped is not None: # data
        h5_file_name=f'SNR_Indi_A{Station}_R{Run}.h5'
    else:
        h5_file_name=f'SNR_Indi_Sim_wo_cut.h5'
    hf = h5py.File(h5_file_name, 'w')
    hf.create_dataset('wf', data=dat_all, compression="gzip", compression_opts=9)

    # fft conversion
    freq = freq_pad_maker()[0]
    dat_all = fft_maker(dat_all)

    #hf.create_dataset('fft', data=dat_all, compression="gzip", compression_opts=9)

    # remove sc
    if Ped is not None: # data
        sc_arr = sc_loader(Station, Run, oneside = False, db_scale = False)
        dat_all /= sc_arr[:,:,np.newaxis]
        del sc_arr 
    else: # sim.
        pass

    #hf.create_dataset('fft_wo_sc', data=dat_all, compression="gzip", compression_opts=9)

    # load psd. v^2/hz scale
    psd_all = ntot_loader(CPath, Station, oneside = False, dbmphz_scale = False)

    # load temp
    temp_all = temp_loader(CPath, 'EM')[0][:,:,0]

    # band pass filter
    dat_all = band_square(freq,dat_all)
    temp_all = band_square(freq,temp_all)
    psd_all = band_square(freq,psd_all)
    del freq

    # matched filter
    snr_max, snr_max_lag = mf_max_snr_lag(dat_all, temp_all, psd_all, hilb = True)
    del dat_all, temp_all, psd_all
    """
    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    if Ped is not None: # data
        h5_file_name=f'SNR_Indi_A{Station}_R{Run}.h5'
    else:
        h5_file_name=f'SNR_Indi_Sim.h5'
    hf = h5py.File(h5_file_name, 'w')   
    """
    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('snr_max_lag', data=snr_max_lag, compression="gzip", compression_opts=9)
    hf.create_dataset('snr_max', data=snr_max, compression="gzip", compression_opts=9)

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) < 3 or len (sys.argv) > 4:
        Usage = """
    This is designed to analyze all events in the data or simulation. Need to choose specific run or sim file.
    
    If it is data,
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Rayl/>
    
    If it is simulation,
    Usage = python3 %s
    <Raw file ex)/data/user/mkim/Rayl_tuning/AraOut.setup_A2_random_N_P-3_N2000_inice_N_est_rayl.txt.run103.h5>
    <Output path ex)/data/user/mkim/Rayl_tuning/>
        """ %(sys.argv[0], sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)
   
    # argv
    data=str(sys.argv[1])

    if data[-4:] == 'root':
        ped=str(sys.argv[2])
        output=str(sys.argv[3])
    
    elif data[-2:] == 'h5':
        ped=None
        output=str(sys.argv[2])
  
    matched_filter_indi_snr(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
