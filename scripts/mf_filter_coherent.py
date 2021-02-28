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
from tools.temp import temp_loader
from tools.mf import soft_psd_maker
from tools.mf import evt_snr_maker

def ms_filter(Data, Ped, Station, Run, Output, DMode, CPath = curr_path, Sel_evt_soft = None, Sel_evt = None):

    if DMode == 'normal':
        print('DMode is normal! Final result will be only event-wise SNR!')
        del Sel_evt_soft, Sel_evt
    elif DMode == 'debug' and Sel_evt_soft is not None and Sel_evt is not None:
        print('DMode is debug! It will save all middle progress by h5 and png!')
        print(f'Selected events are {Sel_evt_soft}(Soft) and {Sel_evt}')
    else:
        print('DMode is not set. Choose 1) normal or 2) debug w/ events')
        sys.exit(1)

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
    freq, freq_w = freq_pad_maker(time_pad_len, time_width_s)
    del time_width_s

    # template loader
    temp_vol, num_theta, theta_w, peak_index = temp_loader(CPath, 'EM')

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)

    # psd maker
    if DMode == 'normal':

        # psd maker
        soft_psd = soft_psd_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq # freq
                                    , num_theta # theta
                                    , DMode) # argv

        # event-wise snr maker
        evt_w_snr_v, evt_w_snr_h, evts_num, trig = evt_snr_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq, freq_w # freq
                                    , soft_psd # psd
                                    , temp_vol, num_theta, theta_w, peak_index # temp
                                    , DMode, Station, CPath) # argv
        del soft_psd, CPath, ROOT, file, eventTree, rawEvent, num_events, calibrator, qual, num_Antennas, time_pad_len, time_width_ns, time_pad_i, time_pad_f, freq, freq_w, temp_vol, num_theta, theta_w, peak_index, bad_ant_index, Ndf

        # create output file
        os.chdir(Output)

        h5_file_name=f'Evt_Wise_SNR_A{Station}_R{Run}.h5'
        hf = h5py.File(h5_file_name, 'w')
        del Station, Run

        #saving result
        hf.create_dataset('evt_w_snr_v', data=evt_w_snr_v, compression="gzip", compression_opts=9)
        hf.create_dataset('evt_w_snr_h', data=evt_w_snr_h, compression="gzip", compression_opts=9)
        hf.create_dataset('evts_num', data=evts_num, compression="gzip", compression_opts=9)
        hf.create_dataset('trig', data=trig, compression="gzip", compression_opts=9)
        del evt_w_snr_v, evt_w_snr_h, evts_num, trig

        hf.close()
        del hf

    elif DMode == 'debug' and Sel_evt_soft is not None and Sel_evt is not None:

        # selected event
        #Sel_evt_soft = 9
        #Sel_evt = 10 # Soft
        #Sel_evt = 11 # Cal
        #Sel_evt = 13 # RF

        # psd maker
        soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band = soft_psd_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq # freq
                                    , num_theta #theta
                                    , DMode, Station, Run, Output, Sel_evt_soft) # argv

 
        # event-wise snr maker
        evt_w_snr, evt_w_snr_v, evt_w_snr_h, evts_num, trig, trig_index, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy, snr_wf_copy_1, snr_wf_r_max_copy, v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range, time_pad, mov_t, pad_t, peak_w, v_opt_angle, h_opt_angle = evt_snr_maker(ROOT, eventTree, rawEvent, num_events, calibrator, qual # ara root
                                    , num_Antennas, bad_ant_index, time_width_ns, Ndf # known config
                                    , time_pad_len, time_pad_i, time_pad_f # time
                                    , freq, freq_w # freq
                                    , soft_psd # psd
                                    , temp_vol, num_theta, theta_w, peak_index # temp
                                    , DMode, Station, CPath, Run, Output, Sel_evt) # argv
        del CPath, ROOT, file, eventTree, rawEvent, num_events, calibrator, qual, num_Antennas, time_pad_len, time_width_ns, time_pad_i, time_pad_f, freq_w, temp_vol, num_theta, peak_index, Ndf
 
        # create output file
        os.chdir(Output)

        h5_file_name=f'Evt-Wise_SNR_A{Station}_R{Run}_debug.h5'
        hf = h5py.File(h5_file_name, 'w')
        del Station, Run

        #saving result
        g0 = hf.create_group('Station_info')
        g0.create_dataset('Bad_ant', data=np.array([bad_ant_index]), compression="gzip", compression_opts=9)
        del g0, bad_ant_index

        g1 = hf.create_group('PSD')
        g1.create_dataset('Freq', data=freq, compression="gzip", compression_opts=9)
        g1.create_dataset('Time', data=time_pad, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD', data=soft_psd_wo_band, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_PSD_band', data=soft_psd, compression="gzip", compression_opts=9)
        g1.create_dataset('Soft_evt_num', data=np.array([soft_evt_num]), compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_WF_evt{Sel_evt_soft}', data=soft_wf, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_FFT_evt{Sel_evt_soft}', data=soft_fft, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_FFT_band_evt{Sel_evt_soft}', data=soft_fft_band, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_PSD_evt{Sel_evt_soft}', data=soft_indi_psd, compression="gzip", compression_opts=9)
        g1.create_dataset(f'Soft_PSD_band_evt{Sel_evt_soft}', data=soft_indi_psd_band, compression="gzip", compression_opts=9)
        del g1, Sel_evt_soft, soft_psd, soft_psd_wo_band, soft_evt_num, soft_wf, soft_fft, soft_fft_band, soft_indi_psd, soft_indi_psd_band 

        g2 = hf.create_group('Event_wise_SNR_indi')
        g2.create_dataset('Trig_index', data=np.array([trig_index]), compression="gzip", compression_opts=9)
        g2.create_dataset('Freq', data=freq, compression="gzip", compression_opts=9)
        g2.create_dataset('Time', data=time_pad, compression="gzip", compression_opts=9)
        g2.create_dataset(f'WF_evt{Sel_evt}', data=ant_arr_copy, compression="gzip", compression_opts=9)
        g2.create_dataset(f'FFT_evt{Sel_evt}', data=ant_arr_fft, compression="gzip", compression_opts=9)
        g2.create_dataset(f'FFT_band_evt{Sel_evt}', data=ant_arr_fft_band, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_time', data=pad_t, compression="gzip", compression_opts=9)
        g2.create_dataset('Peak_search_width', data=np.array([peak_w]), compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_evt{Sel_evt}', data=snr_wf_copy, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_evt{Sel_evt}_wo_tale', data=snr_wf_copy_1, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_roll_max_evt{Sel_evt}', data=snr_wf_r_max_copy, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_time', data=mov_t, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_opt_Vpol', data=v_opt_angle, compression="gzip", compression_opts=9)
        g2.create_dataset('SNR_shift_opt_Hpol', data=h_opt_angle, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_evt{Sel_evt}_Vpol', data=v_match, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_evt{Sel_evt}_Hpol', data=h_match, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_sum_evt{Sel_evt}_Vpol', data=v_sum_match, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_sum_evt{Sel_evt}_Hpol', data=h_sum_match, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_sum_count_evt{Sel_evt}_Vpol', data=v_sum_match_01, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_sum_count_evt{Sel_evt}_Hpol', data=h_sum_match_01, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_avg_evt{Sel_evt}_Vpol', data=v_avg_match, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_shift_roll_max_avg_evt{Sel_evt}_Hpol', data=h_avg_match, compression="gzip", compression_opts=9)
        g2.create_dataset('Nadir_range', data=nadir_range, compression="gzip", compression_opts=9)
        g2.create_dataset('Phi_range', data=phi_range, compression="gzip", compression_opts=9)
        g2.create_dataset('Grid_width', data=np.array([theta_w]), compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_2d_evt{Sel_evt}_Vpol', data=evt_snr_v_sky_2d, compression="gzip", compression_opts=9)
        g2.create_dataset(f'SNR_2d_evt{Sel_evt}_Hpol', data=evt_snr_h_sky_2d, compression="gzip", compression_opts=9)
        del g2, Sel_evt, trig_index, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy, snr_wf_copy_1, snr_wf_r_max_copy, v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range, time_pad, mov_t, pad_t, freq, peak_w, theta_w, v_opt_angle, h_opt_angle 

        gf = hf.create_group('Event_wise_SNR')
        gf.create_dataset('evt_w_snr', data=evt_w_snr, compression="gzip", compression_opts=9)
        gf.create_dataset('evt_w_snr_v', data=evt_w_snr_v, compression="gzip", compression_opts=9)
        gf.create_dataset('evt_w_snr_h', data=evt_w_snr_h, compression="gzip", compression_opts=9)
        gf.create_dataset('evts_num', data=evts_num, compression="gzip", compression_opts=9)
        gf.create_dataset('trig', data=trig, compression="gzip", compression_opts=9)
        del gf, evt_w_snr, evt_w_snr_v, evt_w_snr_h, evts_num, trig

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
    if len (sys.argv) !=7 and len (sys.argv) !=9:
        Usage = """
    This is designed to analyze all events in the run. You have to choose specific run.
    Depending on DMode, It will save just event-wise SNR or all middle step information.
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Station ex)2>
    <Run ex)2898>
    <Output path ex)/data/user/mkim/OMF_filter/ARA02/>
    <DMode ex) normal or debug>
    if DMode is debug, 
        <Sel_evt_soft ex) 9>
        <Sel_evt ex) 10(Soft) 11(Cal) or 13(RF)>
        """ %(sys.argv[0])
        print(Usage)
        del Usage
        sys.exit(1)

    Data=str(sys.argv[1])
    Ped=str(sys.argv[2])
    Station=int(sys.argv[3])
    Run=int(sys.argv[4])
    Output=str(sys.argv[5])
    DMode=str(sys.argv[6])
    if DMode == 'debug':
        sel_soft = int(sys.argv[7])
        sel = int(sys.argv[8])
        ms_filter(Data, Ped, Station, Run, Output, DMode, CPath = curr_path+'/..', Sel_evt_soft = sel_soft, Sel_evt = sel)
        del sel_soft, sel

    else:
        ms_filter(Data, Ped, Station, Run, Output, DMode, CPath = curr_path+'/..')

del curr_path













    
