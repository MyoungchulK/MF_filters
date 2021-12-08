import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.run import data_info_reader
from tools.run import evt_num_loader
from tools.chunk import wf_collector_dat
from tools.fft import freq_pad_maker
from tools.fft import fft_maker
from tools.fit import rayleigh_fit
from tools.fit import rayleigh_fit_complicate
from tools.fft import psd_maker
from tools.sc import sc_maker

def rayl_fit(CPath = curr_path, Data = None, Ped = None, Output = None):

    # collecting WF and info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    evt_entry, wf_len_all = evt_num_loader(Station, Run, trig_set = 2, qual_set = 0, add_info = 'wf_len_all')
    
    # collecting wf
    wf_all = wf_collector_dat(Data, Ped, Station, evt_entry)[0]

    # fft conversion
    freq, df = freq_pad_maker(oneside = True, dfreq = True)
    wf_all = fft_maker(wf_all, oneside = True, symmetry = True, absolute = True, ortho_norm = wf_len_all)

    # rayleigh fitting w/ sc
    #mu_w_sc = rayleigh_fit(wf_all)
    mu_w_sc, chi2_w_sc, hist_arr_w_sc, hist_err_arr_w_sc, pdf_arr_w_sc, bin_range_w_sc, bin_center_w_sc = rayleigh_fit_complicate(wf_all, binning = 50, save_mu = True, save_chi2 = True, save_pdf = True, save_hist = True, save_hist_err = True, save_bin = True)   
 
    # psd conversion
    psd = psd_maker(mu_w_sc/2, df, oneside = True, symmetry = True, dbm_per_hz = True)

    # sc making
    sc = sc_maker(psd, CPath, Station, rfft = True, fft_norm = True)

    # remove sc
    wf_all_wo_sc = wf_all / sc[:,:,np.newaxis]
    
    # rayleigh fitting w/ sc
    #mu_wo_sc = rayleigh_fit(wf_all_wo_sc)
    mu_wo_sc, chi2_wo_sc, hist_arr_wo_sc, hist_err_arr_wo_sc, pdf_arr_wo_sc, bin_range_wo_sc, bin_center_wo_sc = rayleigh_fit_complicate(wf_all_wo_sc, binning = 50, save_mu = True, save_chi2 = True, save_pdf = True, save_hist = True, save_hist_err = True, save_bin = True)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Rayl_A{Station}_R{Run}.h5'
    hf = h5py.File(h5_file_name, 'w')   

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('evt_entry', data=evt_entry, compression="gzip", compression_opts=9)
    hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
    hf.create_dataset('mu_w_sc', data=mu_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('mu_wo_sc', data=mu_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('sc', data=sc, compression="gzip", compression_opts=9)
    hf.create_dataset('psd', data=psd, compression="gzip", compression_opts=9)    

    hf.create_dataset('chi2_w_sc', data=chi2_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('hist_arr_w_sc', data=hist_arr_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('hist_err_arr_w_sc', data=hist_err_arr_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('pdf_arr_w_sc', data=pdf_arr_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('bin_range_w_sc', data=bin_range_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('bin_center_w_sc', data=bin_center_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('chi2_wo_sc', data=chi2_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('hist_arr_wo_sc', data=hist_arr_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('hist_err_arr_wo_sc', data=hist_err_arr_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('pdf_arr_wo_sc', data=pdf_arr_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('bin_range_wo_sc', data=bin_range_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('bin_center_wo_sc', data=bin_center_wo_sc, compression="gzip", compression_opts=9)

    hf.create_dataset('wf_len_all', data=wf_len_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_all', data=wf_all, compression="gzip", compression_opts=9)
    hf.create_dataset('wf_all_wo_sc', data=wf_all_wo_sc, compression="gzip", compression_opts=9)
    del wf_all, wf_all_wo_sc, psd, wf_len_all, evt_entry
    del freq, sc, Station, Run, Config, Year, Month, Date, mu_w_sc, mu_wo_sc, df
    #del chi2_w_sc, hist_arr_w_sc, pdf_arr_w_sc, bin_range_w_sc, bin_center_w_sc, chi2_wo_sc, hist_arr_wo_sc, pdf_arr_wo_sc, bin_range_wo_sc, bin_center_wo_sc

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    file_size = np.round(os.path.getsize(Output+h5_file_name)/1204/1204,2)
    print('file size is', file_size, 'MB')
    del Output, h5_file_name, file_size
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
    
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    # argv
    data=str(sys.argv[1])
    ped=str(sys.argv[2])
    output=str(sys.argv[3])

    rayl_fit(CPath = curr_path+'/..', Data = data, Ped = ped, Output = output)

del curr_path













    
