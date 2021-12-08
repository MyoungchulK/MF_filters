import numpy as np
import os, sys
import h5py

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_root import ara_root_lib
from tools.ara_root import ara_raw_to_qual
import tools.antenna as ant
from tools.wf import interpolation_bin_width
from tools.wf import time_pad_maker
from tools.fft import freq_pad_maker
from tools.fft import psd_maker_chunk
from tools.mf import Band_Square
from tools.run import data_info_reader
from tools.temp import temp_loader
from tools.chunk import wf_collector

def ms_filter(Data, Ped, Output, CPath = curr_path, DMode = False, Sel_evt_soft = None):

    if DMode == True:
        print('Debug mode! It will save all middle progress by h5 and png!')
        if Sel_evt_soft is not None:
            print(f'Selected event is {Sel_evt_soft}')
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

    # known configuration. Probably can call from actual data file through the AraRoot in future....
    #antenna
    num_Antennas = ant.antenna_info()[2]
    # masked antenna
    bad_ant_index = ant.bad_antenna(Station, Run)
    # interpolation time width
    time_width_ns, time_width_s, Ndf =interpolation_bin_width()

    # make wf pad
    time_pad_len, time_pad_i, time_pad_f = time_pad_maker(time_width_ns)[1:]

    # make freq pad
    freq = freq_pad_maker(time_pad_len, time_width_s)[0]
    del time_width_s

    # template loader
    num_theta = temp_loader(CPath, 'EM')[1]

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)

    # collecting WF (chunk analysis way)
    if DMode == False:
        soft_wf_all, soft_int_wf_len_all = wf_collector(ROOT, eventTree, rawEvent, 0, num_events, calibrator, qual # ara root
                                                        , num_Antennas, time_width_ns # known config
                                                        , time_pad_len, time_pad_i, time_pad_f # time
                                                        , trig_set = 2, qual_set = 1, wf_len = True)
    else:
        soft_wf_all, soft_int_wf_len_all, soft_evt_num = wf_collector(ROOT, eventTree, rawEvent, 0, num_events, calibrator, qual # ara root
                                                        , num_Antennas, time_width_ns # known config
                                                        , time_pad_len, time_pad_i, time_pad_f # time
                                                        , trig_set = 2, qual_set = 1, wf_len = True, evt_info = True)

        # checking selected event
        if Sel_evt_soft is not None:
            try:
                evt_index = np.where(soft_evt_num == Sel_evt_soft)[0][0]
            except IndexError:
                print(f'event#{Sel_evt_soft} is not software trigger event!')
                Sel_evt_soft = np.random.choice(soft_evt_num,1)[0]
                evt_index = np.where(soft_evt_num == Sel_evt_soft)[0][0]
                print(f'Re-selected event is {Sel_evt_soft}(Soft)')
        else:
            Sel_evt_soft = np.random.choice(soft_evt_num,1)[0]
            evt_index = np.where(soft_evt_num == Sel_evt_soft)[0][0]
            print(f'Selected event is {Sel_evt_soft}(Soft)')

        # count number of soft event
        soft_evt_num = len(soft_evt_num)

        # debugging plot
        from tools.debug import psd_indi_debug
        soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band = psd_indi_debug(Station, Run, Output, Sel_evt_soft
                            , time_pad_len, time_pad_i, time_pad_f, time_width_ns, Ndf
                            , freq
                            , num_theta
                            , soft_wf_all[:,:,evt_index], soft_int_wf_len_all[:,evt_index]) 

    # making psd
    print('Making PSD starts!')
    soft_psd_all = psd_maker_chunk(soft_wf_all, Ndf, time_pad_len, soft_int_wf_len_all)
    del soft_wf_all, soft_int_wf_len_all

    # makeing mean and std
    soft_psd_mean = np.nanmean(soft_psd_all,axis=2)
    soft_psd_std = np.nanstd(soft_psd_all,axis=2)
    del soft_psd_all

    if DMode == True:
        # debugging plot
        from tools.debug import psd_debug
        psd_copy = psd_debug(Station, Run, Output
                , freq
                , num_theta
                , soft_psd_mean, Sel_evt_soft)

    # band pass filter
    soft_psd_mean_band = np.copy(soft_psd_mean)
    soft_psd_mean_band = Band_Square(freq, np.repeat(soft_psd_mean_band[:,:,np.newaxis], num_theta, axis=2))
    print('PSD making is done!')

    # create output file
    os.chdir(Output)
    h5_file_name=f'Soft_PSD_A{Station}_R{Run}.h5'
    if DMode == True:
        h5_file_name=f'Soft_PSD_A{Station}_R{Run}_debug.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    hf.create_dataset('Config', data=np.array([Station, Run, Config, Year, Month, Date]), compression="gzip", compression_opts=9)
    hf.create_dataset('Bad_ant', data=np.array([bad_ant_index]), compression="gzip", compression_opts=9)    
    hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_psd', data=soft_psd_mean, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_psd_std', data=soft_psd_std, compression="gzip", compression_opts=9)
    hf.create_dataset('soft_psd_band', data=soft_psd_mean_band, compression="gzip", compression_opts=9)
    if DMode == True:
        hf.create_dataset('Time', data=np.arange(time_pad_i, time_pad_f+time_width_ns, time_width_ns), compression="gzip", compression_opts=9)
        hf.create_dataset('Soft_evt_num', data=np.array([soft_evt_num]), compression="gzip", compression_opts=9)
        hf.create_dataset('Sel_evt_soft', data=np.array([Sel_evt_soft]), compression="gzip", compression_opts=9)
        hf.create_dataset(f'Soft_WF_evt{Sel_evt_soft}', data=soft_wf, compression="gzip", compression_opts=9)
        hf.create_dataset(f'Soft_FFT_evt{Sel_evt_soft}', data=soft_fft, compression="gzip", compression_opts=9)
        hf.create_dataset(f'Soft_FFT_band_evt{Sel_evt_soft}', data=soft_fft_band, compression="gzip", compression_opts=9)
        hf.create_dataset(f'Soft_PSD_evt{Sel_evt_soft}', data=soft_indi_psd, compression="gzip", compression_opts=9)
        hf.create_dataset(f'Soft_PSD_band_evt{Sel_evt_soft}', data=soft_indi_psd_band, compression="gzip", compression_opts=9)         
    hf.close()
    #del hf, freq, soft_psd, soft_psd_wo_band, Config

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
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/PSD/>
    if you want debug plot,
        <DMode ex) normal or debug>
        if you want specitic event,
            <Sel_evt_soft ex) 9>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Output=str(sys.argv[3])
    if len (sys.argv) == 5 and str(sys.argv[4]) == 'debug':
        dmode=True
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode)
    elif len (sys.argv) == 6 and str(sys.argv[4]) == 'debug':
        dmode=True
        sel_soft = int(sys.argv[5])
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode, Sel_evt_soft = sel_soft)
        del sel_soft
    else:
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..')

del curr_path













    
