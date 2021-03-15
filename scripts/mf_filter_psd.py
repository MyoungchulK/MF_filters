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
from tools.mf import soft_psd_maker
from tools.run import data_info_reader
from tools.temp import temp_loader

def ms_filter(Data, Ped, Output, CPath = curr_path, DMode = 'normal', Sel_evt_soft = None):

    if DMode == 'normal':
        print('DMode is normal! Final result will be only noise PSD!')
        del Sel_evt_soft
    elif DMode == 'debug'and Sel_evt_soft is None:
        print('DMode is debug! It will save all middle progress by h5 and png!')
    elif DMode == 'debug' and Sel_evt_soft is not None:
        print('DMode is debug! It will save all middle progress by h5 and png!')
        print(f'Selected events are {Sel_evt_soft}(Soft)')
    else:
        print('DMode is not set. Choose 1) normal or 2) debug w/ events')
        sys.exit(1)

    # read data info
    Station, Run, Config = data_info_reader(Data)[:3]

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

    # psd maker
    if DMode == 'normal':

        # psd maker
        soft_psd, soft_psd_wo_band = soft_psd_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq # freq
                                    , num_theta # theta
                                    , DMode) # argv
        del CPath, ROOT, file, eventTree, rawEvent, num_events, calibrator, qual, num_Antennas, time_pad_len, time_width_ns, time_pad_i, time_pad_f, bad_ant_index, Ndf, num_theta

        # create output file
        os.chdir(Output)

        h5_file_name=f'Soft_PSD_A{Station}_R{Run}.h5'
        hf = h5py.File(h5_file_name, 'w')
        del Station, Run

        #saving result
        hf.create_dataset('Config', data=np.array([Config]), compression="gzip", compression_opts=9)
        hf.create_dataset('freq', data=freq, compression="gzip", compression_opts=9)
        hf.create_dataset('soft_psd', data=soft_psd_wo_band, compression="gzip", compression_opts=9)
        hf.create_dataset('soft_psd_band', data=soft_psd, compression="gzip", compression_opts=9)
        hf.close()
        del hf, freq, soft_psd, soft_psd_wo_band, Config

    elif DMode == 'debug':

        # selected event
        #Sel_evt_soft = 9

        # psd maker
        soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band, Sel_evt_soft = soft_psd_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq # freq
                                    , num_theta #theta
                                    , DMode, Station, Run, Output, Sel_evt_soft) # argv
        del CPath, ROOT, file, eventTree, rawEvent, num_events, calibrator, qual, num_Antennas, time_pad_len, Ndf, num_theta
 
        # create output file
        os.chdir(Output)

        h5_file_name=f'Soft_PSD_A{Station}_R{Run}_debug.h5'
        hf = h5py.File(h5_file_name, 'w')
        del Station, Run

        #saving result
        g0 = hf.create_group('Run_info')
        g0.create_dataset('Bad_ant', data=np.array([bad_ant_index]), compression="gzip", compression_opts=9)
        g0.create_dataset('Config', data=np.array([Config]), compression="gzip", compression_opts=9)
        del g0, bad_ant_index, Config

        g1 = hf.create_group('PSD')
        g1.create_dataset('Freq', data=freq, compression="gzip", compression_opts=9)
        g1.create_dataset('Time', data=np.arange(time_pad_i, time_pad_f+time_width_ns, time_width_ns), compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD', data=soft_psd_wo_band, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD_band', data=soft_psd, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_evt_num', data=np.array([soft_evt_num]), compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_WF_evt{Sel_evt_soft}', data=soft_wf, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_FFT_evt{Sel_evt_soft}', data=soft_fft, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_FFT_band_evt{Sel_evt_soft}', data=soft_fft_band, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_PSD_evt{Sel_evt_soft}', data=soft_indi_psd, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_PSD_band_evt{Sel_evt_soft}', data=soft_indi_psd_band, compression="gzip", compression_opts=9)
        del g1, Sel_evt_soft, soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band, freq, time_width_ns, time_pad_i, time_pad_f 

        hf.close()
        del hf        

    else:

        print('DMode is not set. Choose 1) normal or 2) debug w/ events')    
        sys.exit(1)  
    del DMode

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
        dmode=str(sys.argv[4])
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode)
    elif len (sys.argv) == 6 and str(sys.argv[4]) == 'debug':
        dmode=str(sys.argv[4])
        sel_soft = int(sys.argv[5])
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..', DMode = dmode, Sel_evt_soft = sel_soft)
        del sel_soft

    else:
        ms_filter(Data, Ped, Output, CPath = curr_path+'/..')

del curr_path













    
