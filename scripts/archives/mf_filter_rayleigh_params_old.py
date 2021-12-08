import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_root import ara_root_lib
from tools.ara_root import ara_raw_to_qual
from tools.fft import freq_pad_maker
from tools.fft import fft_maker
from tools.run import data_info_reader
from tools.chunk import wf_collector
from tools.fit import rayleigh_fit
from tools.fft import psd_maker
from tools.sc import sc_maker

def rayl_fit(Data, Ped, Output, CPath = curr_path, DMode = False, Sel_evt = None):

    if DMode == True:
        print('Debug mode! It will save all middle progress by h5 and png!')
        if Sel_evt is not None:
            print(f'Selected event is {Sel_evt}')
    elif DMode == False:
        print('Normal mode! Final result will be only noise PSD!')
    else:
        print('DMode is not set. Choose 1) normal or 2) debug w/ events')
        sys.exit(1)
    
    # read data info
    Station, Run, Config, Year, Month, Date = data_info_reader(Data)

    # import root and ara root lib
    ROOT = ara_root_lib()

    # load raw data and process to general quality cut by araroot
    file, eventTree, rawEvent, num_events, calibrator, qual = ara_raw_to_qual(ROOT, Data, Ped, Station)
    del Data, Ped

    # collecting WF
    wf_all, int_wf_len_all, evt_all = wf_collector(ROOT, eventTree, rawEvent, 0, num_events, calibrator, qual # ara root
                            , trig_set = 2, qual_set = 1, wf_len = True, evt_info = True)[:3]
    #del ROOT, file, eventTree, rawEvent, num_events, calibrator, qual

    # fft conversion
    freq = freq_pad_maker(oneside = True)[0]
    fft_all = fft_maker(wf_all, oneside = True, absolute = True)
    del wf_all

    # rayleigh fitting w/ sc
    loc_w_sc, scale_w_sc = rayleigh_fit(fft_all) 

    # psd conversion
    psd = psd_maker(loc_w_sc + scale_w_sc, pad_norm = True, dbm_per_hz = True, int_t_len = np.nanmean(int_wf_len_all, axis = 1))

    # sc making
    sc = sc_maker(psd, CPath, Station)
    del psd

    # remove sc
    fft_all_wo_sc = fft_all / sc[:,:,np.newaxis]
    
    # rayleigh fitting w/ sc
    loc_wo_sc, scale_wo_sc = rayleigh_fit(fft_all_wo_sc)
    if DMode ==True:
        pass
    else:
        del fft_all_wo_sc, fft_all

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
    h5_file_name=f'Rayleigh_Fit_A{Station}_R{Run}.h5'
    if DMode == True:
        h5_file_name=f'Rayleigh_Fit_A{Station}_R{Run}_debug.h5'
    hf = h5py.File(h5_file_name, 'w')   

    #saving result
    hf.create_dataset('config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
    hf.create_dataset('loc_w_sc', data=loc_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('scale_w_sc', data=scale_w_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('loc_wo_sc', data=loc_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('scale_wo_sc', data=scale_wo_sc, compression="gzip", compression_opts=9)
    hf.create_dataset('sc', data=sc, compression="gzip", compression_opts=9)
    if DMode == True:
        hf.create_dataset('fft_all', data=fft_all, compression="gzip", compression_opts=9)
        hf.create_dataset('fft_all_wo_sc', data=fft_all_wo_sc, compression="gzip", compression_opts=9)
        del fft_all, fft_all_wo_sc
    del freq, loc_w_sc, scale_w_sc, loc_wo_sc, scale_wo_sc, sc, Station, Run, Config, Year, Month, Date, evt_all

    hf.close()

    print(f'output is {Output}{h5_file_name}')
    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=4 and len (sys.argv) !=5 and len (sys.argv) !=6:
        Usage = """
    This is designed to analyze all events in the run. You have to choose specific run.
    Depending on DMode, It will save just psd or all middle step information.
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/Rayl/>
    if you want debug plot,
        <DMode ex) normal or debug>
        if you want specitic event,
            <Sel_evt ex) 9>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Output=str(sys.argv[3])
    if len (sys.argv) == 5 and str(sys.argv[4]) == 'debug':
        dmode=True
        rayl_fit(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode)
    elif len (sys.argv) == 6 and str(sys.argv[4]) == 'debug':
        dmode=True
        sel_event = int(sys.argv[5])
        rayl_fit(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode, Sel_evt = sel_event)
        del sel_event
    else:
        rayl_fit(Data, Ped, Output, CPath = curr_path+'/..')

del curr_path













    
