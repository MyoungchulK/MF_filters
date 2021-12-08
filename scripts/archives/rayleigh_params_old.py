import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.run import evt_num_loader
#from tools.chunk import wf_collector_dat
#from tools.chunk import wf_collector_sim
from tools.fft import freq_pad_maker
from tools.fft import fft_maker
from tools.fit import rayleigh_fit
from tools.fft import psd_maker
from tools.sc import sc_maker

def rayl_fit(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting WF and info
    if Ped is not None: # data
        Station, Run, Config, Year, Month, Date = data_info_reader(Data)

        evt_entry = evt_num_loader(Station, Run, trig_set = 0, qual_set = 0, evt_entry = None, act_evt = None)
        print(evt_entry)

        #wf_all, wf_len_all = wf_collector_dat(Data, Ped, Station, trig_set = 2, qual_set = 1, wf_dat = True, wf_len = True, Year = Year)[:2] 
    else: # sim.
        Station = 2
        Run = -1
        Config = -1
        Year = -1
        Month = -1
        Date = -1
        dat_all, dat_int_len = wf_collector_sim(Data, wf_len = True)[:2]
    """
    # fft conversion
    freq = freq_pad_maker(oneside = True)[0]
    wf_all = fft_maker(wf_all, oneside = True, symmetry = True, VpHz = True, absolute = True, pad_norm = wf_len_all, pad_renorm = 247.5)

    # rayleigh fitting w/ sc
    mu_w_sc = rayleigh_fit(wf_all, save_mu = True)[0]
    
    # psd conversion
    psd = psd_maker(mu_w_sc/2, oneside = True, symmetry = True, dbm_per_hz = True)

    # sc making
    sc = sc_maker(psd, CPath, Station, rfft = True, fft_norm = False)

    # remove sc
    wf_all_wo_sc = wf_all / sc[:,:,np.newaxis]
    
    # rayleigh fitting w/ sc
    mu_wo_sc = rayleigh_fit(wf_all_wo_sc, save_mu = True)[0]

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    if Ped is not None: # data
        h5_file_name=f'Rayleigh_Fit_A{Station}_R{Run}.h5'
    else:
        h5_file_name=f'Rayleigh_Fit_A{Station}_Sim.h5'
    hf = h5py.File(h5_file_name, 'w')   

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
    hf.create_dataset('mu_w_sc', data=mu_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('chi2_w_sc', data=chi2_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('hist_arr_w_sc', data=hist_arr_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('hist_err_arr_w_sc', data=hist_err_arr_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('pdf_arr_w_sc', data=pdf_arr_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('bin_range_w_sc', data=bin_range_w_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('bin_center_w_sc', data=bin_center_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('mu_wo_sc', data=mu_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('chi2_wo_sc', data=chi2_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('hist_arr_wo_sc', data=hist_arr_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('hist_err_arr_wo_sc', data=hist_err_arr_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('pdf_arr_wo_sc', data=pdf_arr_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('bin_range_wo_sc', data=bin_range_wo_sc, compression="gzip", compression_opts=9)
    #hf.create_dataset('bin_center_wo_sc', data=bin_center_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('sc', data=sc, compression="gzip", compression_opts=9)
    hf.create_dataset('psd', data=psd, compression="gzip", compression_opts=9)    

    #hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)
    #hf.create_dataset('wf_all', data=ef_all, compression="gzip", compression_opts=9)
    #hf.create_dataset('wf_all_wo_sc', data=ef_all_wo_sc, compression="gzip", compression_opts=9)
    del wf_all, wf_all_wo_sc, psd, wf_len_all
    del freq, sc, Station, Run, Config, Year, Month, Date, mu_w_sc, mu_wo_sc
    #del chi2_w_sc, hist_arr_w_sc, pdf_arr_w_sc, bin_range_w_sc, bin_center_w_sc, chi2_wo_sc, hist_arr_wo_sc, pdf_arr_wo_sc, bin_range_wo_sc, bin_center_wo_sc

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    del Output, h5_file_name
    print('done!')
    """
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

    rayl_fit(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
