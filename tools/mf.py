import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import maximum_filter1d
from tqdm import tqdm

# custom lib
#from tools.array import arr_1d
from tools.array import arr_2d
from tools.array import arr_3d
from tools.ara_root import useful_evt_maker
from tools.ara_root import trig_checker
from tools.wf import station_pad_soft
from tools.wf import station_pad
#from tools.wf import station_pad_w_index
#from tools.plot import plot_16_log_theta
#from tools.plot import plot_16
#from tools.plot import plot_16_3
#from tools.plot import plot_16_overlap
#from tools.plot import plot_1
#from tools.plot import sky_map
#from tools.plot import hist_map
from tools.arr_table import table_loader

def off_pad_maker(pad_len, dt):

    off_pad = np.arange(0,pad_len,1) * dt

    return off_pad, len(off_pad), off_pad[0], off_pad[-1]

def Band_Square(freq,band_amp):
    
    #Bothpol
    band_amp[(freq>=430e6) & (freq<=480e6)]=1e-50
    band_amp[(freq<=-430e6) & (freq>=-480e6)]=1e-50
    
    #Vpol
    band_amp[(freq>=-150e6) & (freq<=150e6),:8]=1e-50
    band_amp[(freq>=680e6) | (freq<=-680e6),:8,1]=1e-50
    band_amp[(freq>=665e6) | (freq<=-665e6),:8,2]=1e-50

    #Hpol
    band_amp[(freq>=-217e6) & (freq<=217e6),8:]=1e-50
    band_amp[(freq>=630e6) | (freq<=-630e6),8:,1]=1e-50
    band_amp[(freq>=530e6) | (freq<=-530e6),8:,2]=1e-50

    #Both theta
    band_amp[(freq>=680e6) | (freq<=-680e6),:,0]=1e-50   
 
    return(band_amp)

def Band_Square_debug(freq,amp):

    band_amp = np.copy(amp)

    #Bothpol
    band_amp[(freq>=430e6) & (freq<=480e6)]=1e-50
    band_amp[(freq<=-430e6) & (freq>=-480e6)]=1e-50

    #Vpol
    band_amp[(freq>=-150e6) & (freq<=150e6),:8]=1e-50
    band_amp[(freq>=680e6) | (freq<=-680e6),:8,1]=1e-50
    band_amp[(freq>=665e6) | (freq<=-665e6),:8,2]=1e-50

    #Hpol
    band_amp[(freq>=-217e6) & (freq<=217e6),8:]=1e-50
    band_amp[(freq>=630e6) | (freq<=-630e6),8:,1]=1e-50
    band_amp[(freq>=530e6) | (freq<=-530e6),8:,2]=1e-50

    #Both theta
    band_amp[(freq>=680e6) | (freq<=-680e6),:,0]=1e-50

    return(band_amp)

def psd_maker(vol, dt, i_time_len):

    vol = np.fft.fft(vol, axis=0)
    vol *= dt
    vol *= vol.conjugate()
    vol *= 2
    vol /= dt
    vol /= i_time_len[np.newaxis, :]

    return vol

def OMF(psd, rf_f_w, rf_v, temp_v):
    
    # matched filtering
    conjugation = temp_v * rf_v.conjugate()
    conjugation /= psd
    conjugation = np.abs(hilbert(2*np.fft.ifft(conjugation,axis=0).real, axis = 0))
    #conjugation = np.abs(2*np.fft.ifft(conjugation,axis=0))
    conjugation /= np.sqrt(np.abs(2*np.nansum(temp_v * temp_v.conjugate() / psd,axis=0) * rf_f_w))[np.newaxis, :, :]
    conjugation = conjugation[::-1,:,:]
    conjugation[conjugation<0] = 0 
        
    return conjugation

def soft_psd_maker(R, evtTree, rawEvt, num_evts, cal, q # ara root
                    , num_Ants, bad_ant_i, t_width_ns, t_width_s # known config
                    , t_pad_len, time_pad_i, time_pad_f # time
                    , f # freq
                    , n_theta): #theta

    print('PSD making starts!')

    # array for psd
    psd = arr_2d(t_pad_len, num_Ants, 0, complex)
    num_psd = 0

    # loop over the events
    for event in tqdm(range(num_evts)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        # trigger filter
        if rawEvt.isSoftwareTrigger() == 1:

            # quality cut
            if q.isGoodEvent(usefulEvent) == 1:

                # make padded wf and interpolated wf length
                ant_arr, int_time_len = station_pad_soft(usefulEvent, num_Ants, t_width_ns, time_pad_i, time_pad_f)

                # make psd
                psd += psd_maker(ant_arr, t_width_s, int_time_len)
                num_psd += 1
                del ant_arr, int_time_len

        del usefulEvent

    # averaging
    psd /= num_psd
    print('The number of soft triggered events is', num_psd)
    del num_psd

    # band pass filter
    psd = Band_Square(f, np.repeat(psd[:,:,np.newaxis], n_theta, axis=2))

    # remove bad antenna
    psd[:, bad_ant_i, :] = np.nan

    print('PSD making is done!')

    return psd

def soft_psd_maker_debug(Station, Run, Output, sel_evt # argv
                            , R, evtTree, rawEvt, num_evts, cal, q # ara root
                            , num_Ants, bad_ant_i, t_width_ns, t_width_s # known config
                            , t_pad_len, time_pad_i, time_pad_f # time
                            , f # freq
                            , n_theta): #theta

    from tools.plot import plot_16_log_theta

    print('PSD making starts!')

    # array for psd
    psd = arr_2d(t_pad_len, num_Ants, 0, complex)
    num_psd = 0

    # loop over the events
    for event in tqdm(range(num_evts)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        # trigger filter
        if rawEvt.isSoftwareTrigger() == 1:

            # quality cut
            if q.isGoodEvent(usefulEvent) == 1:

                # make padded wf and interpolated wf length
                ant_arr, int_time_len = station_pad_soft(usefulEvent, num_Ants, t_width_ns, time_pad_i, time_pad_f)

                if event == sel_evt:

                    # wf plot
                    from tools.plot import plot_16
                    ant_arr_copy = np.copy(ant_arr)
                    
                    plot_16(r'Time [ $ns$ ]',r'Amplitude [ $V$ ]','Soft WF, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,np.arange(time_pad_i, time_pad_f+t_width_ns, t_width_ns),ant_arr_copy
                                ,np.round(np.nanmax(np.abs(ant_arr_copy),axis=0),2)
                                ,-220,220
                                ,Output,'Soft_WF_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' Soft WF plot was generated!')
    
                    # fft plot
                    ant_arr_fft = np.fft.fft(ant_arr_copy, axis=0)
                    ant_arr_fft_band = Band_Square_debug(f, np.repeat(ant_arr_fft[:,:,np.newaxis], n_theta, axis=2))

                    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Amplitude [ $V$ ]', 'Soft FFT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,f/1e9,ant_arr_fft
                                ,f/1e9,ant_arr_fft_band[:,:,0]
                                ,f/1e9,ant_arr_fft_band[:,:,1]
                                ,f/1e9,ant_arr_fft_band[:,:,2]
                                ,1e-4,1e2
                                ,Output,'Soft_FFT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' Soft FFT plot was generated!')
            
                else:
                    pass

                # make psd
                psd += psd_maker(ant_arr, t_width_s, int_time_len)

                if event == sel_evt:

                    # psd indi plot
                    indi_psd = psd_maker(ant_arr, t_width_s, int_time_len)
                    indi_psd_band = Band_Square_debug(f, np.repeat(indi_psd[:,:,np.newaxis], n_theta, axis=2))

                    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Power [ $V^2$ ]', 'Soft PSD, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,f/1e9,indi_psd
                                ,f/1e9,indi_psd_band[:,:,0]
                                ,f/1e9,indi_psd_band[:,:,1]
                                ,f/1e9,indi_psd_band[:,:,2]
                                ,1e-15,1e-9
                                ,Output,'Soft_PSD_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' Soft PSD indi plot was generated!')
                
                else:
                    pass

                num_psd += 1
                del ant_arr, int_time_len

        del usefulEvent

    # averaging
    psd /= num_psd
    print('The number of soft triggered events is', num_psd)

    # band pass filter
    psd_band = Band_Square_debug(f, np.repeat(psd[:,:,np.newaxis], n_theta, axis=2))

    # psd plot
    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Power [ $V^2$ ]', 'Soft Avg PSD, A'+str(Station)+', Run'+str(Run)+', Avg of '+str(num_psd)+'Evts'
                    ,f/1e9,psd
                    ,f/1e9,psd_band[:,:,0]
                    ,f/1e9,psd_band[:,:,1]
                    ,f/1e9,psd_band[:,:,2]
                    ,1e-15,1e-9
                    ,Output,'Soft_Avg_PSD_A'+str(Station)+'_Run'+str(Run)+'.png'
                    ,'Soft avg PSD plot was generated!')

    # remove bad antenna
    psd_band[:, bad_ant_i, :] = np.nan

    print('PSD making is done!')

    return psd_band, psd, num_psd, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, indi_psd, indi_psd_band

def evt_snr_maker(Station, CPath # argv
                    , R, evtTree, rawEvt, num_evts, cal, q # ara root
                    , num_Ants, bad_ant_i, t_width_ns # known config
                    , t_pad_l, time_pad_i, time_pad_f # time
                    , f, f_w # freq
                    , n_psd # psd
                    , temp_v, n_theta, theta_w, peak_i): # temp

    print('Event-wise SNR making starts!')

    # table_loader
    peak_w = 50
    mov_i, pad_t_l, p_len_front, p_len_end, ps_len_i = table_loader(CPath, Station, theta_w, peak_w)[:-2]
    del peak_w

    # tale remove range
    tale_i_front= -1*peak_i + p_len_front
    tale_i_end= peak_i + p_len_end

    # 16 pad wf arr
    ant_arr_01 = arr_2d(t_pad_l, num_Ants, 0, int)

    # array for snr
    snr_wf = arr_3d(pad_t_l, num_Ants, n_theta, 0, float)
    snr_wf_01 = arr_3d(pad_t_l, num_Ants, n_theta, 0, int)

    # array for 2d map
    snr_wf_2d_v = arr_3d(mov_i.shape[0], mov_i.shape[2], mov_i.shape[3], 0, float)
    snr_wf_2d_h = np.copy(snr_wf_2d_v)
    snr_wf_2d_01_v = arr_3d(mov_i.shape[0], mov_i.shape[2], mov_i.shape[3], 0, int)
    snr_wf_2d_01_h = np.copy(snr_wf_2d_01_v)

    #array for event-wise snr
    evt_snr = []
    evt_snr_v = []
    evt_snr_h = []
    evt_num = []
    trigger = []
   
    # half antenna number
    half_ant = int(num_Ants/2)

    # loop over the events
    for event in tqdm(range(num_evts)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        # trigger filter
        if rawEvt.isSoftwareTrigger() == 0:

            # quality cut
            if q.isGoodEvent(usefulEvent) == 1:

                # make padded wf with interpolated wf to fft
                ant_arr = station_pad(usefulEvent, num_Ants, t_width_ns, time_pad_i, time_pad_f)

                # 01 antenna array
                ant_arr_01[:] = 0
                ant_arr_01[ant_arr != 0] = 1               
 
                # remove tale step 1
                snr_wf_01[:] = 0
                for n_t in range(n_theta):
                    snr_wf_01[tale_i_front[0][n_t]:-tale_i_end[0][n_t], :half_ant, n_t] = ant_arr_01[:, :half_ant]
                    snr_wf_01[tale_i_front[1][n_t]:-tale_i_end[1][n_t], half_ant:, n_t] = ant_arr_01[:, half_ant:]

                # OMF
                snr_wf[:] = 0
                snr_wf[p_len_front:-p_len_end] = OMF(n_psd, f_w, Band_Square(f, np.repeat(np.fft.fft(ant_arr, axis=0)[:,:,np.newaxis], n_theta, axis=2)), temp_v)
                del ant_arr

                # remove tale step 2
                snr_wf[snr_wf_01 == 0] = 0

                # picking max
                snr_wf = maximum_filter1d(snr_wf, size=ps_len_i, axis=0, mode='constant')

                # 01 array
                snr_wf_01[:] = 0
                snr_wf_01[snr_wf != 0] = 1

                # remove bad antenna
                snr_wf[:, bad_ant_i, :] = 0
                snr_wf_01[:, bad_ant_i, :] = 0
    
                # 2d map array
                snr_wf_2d_v[:] = 0
                snr_wf_2d_h[:] = 0
                snr_wf_2d_01_v[:] = 0
                snr_wf_2d_01_h[:] = 0

                # stack snr into 2d map                
                for half in range(half_ant):

                    snr_wf_2d_v[:,2] += snr_wf[:,half,0][mov_i[:,half,2,:]]
                    snr_wf_2d_v[:,3] += snr_wf[:,half,1][mov_i[:,half,3,:]]
                    snr_wf_2d_v[:,1] += snr_wf[:,half,1][mov_i[:,half,1,:]]
                    snr_wf_2d_v[:,0] += snr_wf[:,half,2][mov_i[:,half,0,:]]
                    snr_wf_2d_v[:,4] += snr_wf[:,half,2][mov_i[:,half,4,:]]
            
                    snr_wf_2d_01_v[:,2] += snr_wf_01[:,half,0][mov_i[:,half,2,:]]
                    snr_wf_2d_01_v[:,3] += snr_wf_01[:,half,1][mov_i[:,half,3,:]]
                    snr_wf_2d_01_v[:,1] += snr_wf_01[:,half,1][mov_i[:,half,1,:]]
                    snr_wf_2d_01_v[:,0] += snr_wf_01[:,half,2][mov_i[:,half,0,:]]
                    snr_wf_2d_01_v[:,4] += snr_wf_01[:,half,2][mov_i[:,half,4,:]]
            
                    snr_wf_2d_h[:,2] += snr_wf[:,half+8,0][mov_i[:,half+8,2,:]]
                    snr_wf_2d_h[:,3] += snr_wf[:,half+8,1][mov_i[:,half+8,3,:]]
                    snr_wf_2d_h[:,1] += snr_wf[:,half+8,1][mov_i[:,half+8,1,:]]
                    snr_wf_2d_h[:,0] += snr_wf[:,half+8,2][mov_i[:,half+8,0,:]]
                    snr_wf_2d_h[:,4] += snr_wf[:,half+8,2][mov_i[:,half+8,4,:]]
            
                    snr_wf_2d_01_h[:,2] += snr_wf_01[:,half+8,0][mov_i[:,half+8,2,:]]
                    snr_wf_2d_01_h[:,3] += snr_wf_01[:,half+8,1][mov_i[:,half+8,3,:]]
                    snr_wf_2d_01_h[:,1] += snr_wf_01[:,half+8,1][mov_i[:,half+8,1,:]]
                    snr_wf_2d_01_h[:,0] += snr_wf_01[:,half+8,2][mov_i[:,half+8,0,:]]
                    snr_wf_2d_01_h[:,4] += snr_wf_01[:,half+8,2][mov_i[:,half+8,4,:]]
        
                evt_snr.append(np.nanmax((snr_wf_2d_v+snr_wf_2d_h) / (snr_wf_2d_01_v+snr_wf_2d_01_h)))
                evt_snr_v.append(np.nanmax(snr_wf_2d_v / snr_wf_2d_01_v))
                evt_snr_h.append(np.nanmax(snr_wf_2d_h / snr_wf_2d_01_h))
                evt_num.append(event)
                trigger.append(trig_checker(rawEvt))

        del usefulEvent

    del mov_i, pad_t_l, p_len_front, p_len_end, ps_len_i, tale_i_front, tale_i_end, ant_arr_01, snr_wf, snr_wf_01, snr_wf_2d_v, snr_wf_2d_01_v, snr_wf_2d_h, snr_wf_2d_01_h, half_ant

    print('Event-wise SNR making is done!')

    return np.asarray(evt_snr), np.asarray(evt_snr_v), np.asarray(evt_snr_h), np.asarray(evt_num), np.asarray(trigger)

def evt_snr_maker_debug(Station, Run, Output, CPath, sel_evt # argv
                        , R, evtTree, rawEvt, num_evts, cal, q # ara root
                        , num_Ants, bad_ant_i, t_width_ns # known config
                        , t_pad_l, time_pad_i, time_pad_f # time
                        , f, f_w # freq
                        , n_psd # psd
                        , temp_v, n_theta, theta_w, peak_i): # temp

    print('Event-wise SNR making starts!')

    # table_loader
    peak_w = 50
    mov_i, pad_t_l, p_len_front, p_len_end, ps_len_i, mov_t, pad_t = table_loader(CPath, Station, theta_w, peak_w)

    # tale remove range
    tale_i_front= -1*peak_i + p_len_front
    tale_i_end= peak_i + p_len_end

    # 16 pad wf arr
    ant_arr_01 = arr_2d(t_pad_l, num_Ants, 0, int)

    # array for snr
    snr_wf = arr_3d(pad_t_l, num_Ants, n_theta, 0, float)
    snr_wf_01 = arr_3d(pad_t_l, num_Ants, n_theta, 0, int)

    # array for 2d map
    snr_wf_2d_v = arr_3d(mov_i.shape[0], mov_i.shape[2], mov_i.shape[3], 0, float)
    snr_wf_2d_h = np.copy(snr_wf_2d_v)
    snr_wf_2d_01_v = arr_3d(mov_i.shape[0], mov_i.shape[2], mov_i.shape[3], 0, int)
    snr_wf_2d_01_h = np.copy(snr_wf_2d_01_v)

    #array for event-wise snr
    evt_snr = []
    evt_snr_v = []
    evt_snr_h = []
    evt_num = []
    trigger = []

    # half antenna number
    half_ant = int(num_Ants/2)

    # loop over the events
    for event in tqdm(range(num_evts)):

        # make a useful event
        usefulEvent = useful_evt_maker(R, evtTree, rawEvt, event, cal)

        # trigger filter
        if rawEvt.isSoftwareTrigger() == 0:

            # quality cut
            if q.isGoodEvent(usefulEvent) == 1:

                # make padded wf with interpolated wf to fft
                ant_arr = station_pad(usefulEvent, num_Ants, t_width_ns, time_pad_i, time_pad_f)

                if event == sel_evt:

                    trig_index = trig_checker(rawEvt)
                    if trig_index == 0:
                        trig_type = 'RF'
                    elif trig_index == 1:
                        trig_type = 'Cal'

                    print('Evt#'+str(event)+' is selected!')
                    print('Trigger type is',trig_type) 

                    # wf plot
                    from tools.plot import plot_16
                    ant_arr_copy = np.copy(ant_arr)
                    
                    plot_16(r'Time [ $ns$ ]',r'Amplitude [ $V$ ]',trig_type+' WF, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,np.arange(time_pad_i, time_pad_f+t_width_ns, t_width_ns),ant_arr_copy
                                ,np.round(np.nanmax(np.abs(ant_arr_copy),axis=0),2)
                                ,time_pad_i,time_pad_f
                                ,Output,trig_type+'_WF_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' '+trig_type+' WF plot was generated!')

                    # fft plot
                    ant_arr_fft = np.fft.fft(ant_arr_copy, axis=0)
                    ant_arr_fft_band = Band_Square_debug(f, np.repeat(ant_arr_fft[:,:,np.newaxis], n_theta, axis=2))

                    from tools.plot import plot_16_log_theta

                    plot_16_log_theta(r'Frequency [ $GHz$ ]',r'Amplitude [ $V$ ]', trig_type+' FFT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,f/1e9,ant_arr_fft
                                ,f/1e9,ant_arr_fft_band[:,:,0]
                                ,f/1e9,ant_arr_fft_band[:,:,1]
                                ,f/1e9,ant_arr_fft_band[:,:,2]
                                ,1e-4,1e2
                                ,Output,trig_type+'_FFT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' '+trig_type+' FFT plot was generated!')
                    
                else:
                    pass

                # 01 antenna array
                ant_arr_01[:] = 0
                ant_arr_01[ant_arr != 0] = 1

                # remove tale step 1
                snr_wf_01[:] = 0
                for n_t in range(n_theta):
                    snr_wf_01[tale_i_front[0][n_t]:-tale_i_end[0][n_t], :half_ant, n_t] = ant_arr_01[:, :half_ant]
                    snr_wf_01[tale_i_front[1][n_t]:-tale_i_end[1][n_t], half_ant:, n_t] = ant_arr_01[:, half_ant:]

                # OMF
                snr_wf[:] = 0
                snr_wf[p_len_front:-p_len_end] = OMF(n_psd, f_w, Band_Square(f, np.repeat(np.fft.fft(ant_arr, axis=0)[:,:,np.newaxis], n_theta, axis=2)), temp_v)
                del ant_arr

                if event == sel_evt:

                    #snr plot
                    from tools.plot import plot_16_3
                    snr_wf_copy = np.copy(snr_wf)

                    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', trig_type+' SNR, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', w/ tale'
                                ,pad_t,snr_wf_copy[:,:,0],np.round(np.nanmax(snr_wf_copy[:,:,0],axis=0),2)
                                ,pad_t,snr_wf_copy[:,:,1],np.round(np.nanmax(snr_wf_copy[:,:,1],axis=0),2)
                                ,pad_t,snr_wf_copy[:,:,2],np.round(np.nanmax(snr_wf_copy[:,:,2],axis=0),2)
                                ,Output,trig_type+'_SNR_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_w_tale.png'
                                ,'Event# '+str(event)+' '+trig_type+' SNR w/ tale plot was generated!')

                else:
                    pass

                # remove tale step 2
                snr_wf[snr_wf_01 == 0] = 0

                if event == sel_evt:

                    #snr plot
                    from tools.plot import plot_16_3
                    snr_wf_copy_1 = np.copy(snr_wf)

                    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', trig_type+' SNR, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', w/o tale'
                                ,pad_t,snr_wf_copy_1[:,:,0],np.round(np.nanmax(snr_wf_copy[:,:,0],axis=0),2)
                                ,pad_t,snr_wf_copy_1[:,:,1],np.round(np.nanmax(snr_wf_copy[:,:,1],axis=0),2)
                                ,pad_t,snr_wf_copy_1[:,:,2],np.round(np.nanmax(snr_wf_copy[:,:,2],axis=0),2)
                                ,Output,trig_type+'_SNR_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_wo_tale.png'
                                ,'Event# '+str(event)+' '+trig_type+' SNR w/o tale plot was generated!')
                
                else:
                    pass

                # picking max
                snr_wf = maximum_filter1d(snr_wf, size=ps_len_i, axis=0, mode='constant')

                if event == sel_evt:

                    #snr roll max plot
                    from tools.plot import plot_16_3
                    snr_wf_r_max_copy = np.copy(snr_wf)

                    plot_16_3(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', trig_type+' Roll Max SNR(pw:'+str(peak_w)+'ns), A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)
                                ,pad_t,snr_wf_r_max_copy[:,:,0],np.round(np.nanmax(snr_wf_r_max_copy[:,:,0],axis=0),2)
                                ,pad_t,snr_wf_r_max_copy[:,:,1],np.round(np.nanmax(snr_wf_r_max_copy[:,:,1],axis=0),2)
                                ,pad_t,snr_wf_r_max_copy[:,:,2],np.round(np.nanmax(snr_wf_r_max_copy[:,:,2],axis=0),2)
                                ,Output,trig_type+'_Roll_Max_SNR_pw'+str(peak_w)+'_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'.png'
                                ,'Event# '+str(event)+' '+trig_type+' Roll Max SNR plot was generated!')

                else:
                    pass
                
                # 01 array
                snr_wf_01[:] = 0
                snr_wf_01[snr_wf != 0] = 1

                # remove bad antenna
                snr_wf[:, bad_ant_i, :] = 0
                snr_wf_01[:, bad_ant_i, :] = 0

                # 2d map array
                snr_wf_2d_v[:] = 0
                snr_wf_2d_h[:] = 0
                snr_wf_2d_01_v[:] = 0
                snr_wf_2d_01_h[:] = 0

                # stack snr into 2d map
                for half in range(half_ant):

                    snr_wf_2d_v[:,2] += snr_wf[:,half,0][mov_i[:,half,2,:]]
                    snr_wf_2d_v[:,3] += snr_wf[:,half,1][mov_i[:,half,3,:]]
                    snr_wf_2d_v[:,1] += snr_wf[:,half,1][mov_i[:,half,1,:]]
                    snr_wf_2d_v[:,0] += snr_wf[:,half,2][mov_i[:,half,0,:]]
                    snr_wf_2d_v[:,4] += snr_wf[:,half,2][mov_i[:,half,4,:]]

                    snr_wf_2d_01_v[:,2] += snr_wf_01[:,half,0][mov_i[:,half,2,:]]
                    snr_wf_2d_01_v[:,3] += snr_wf_01[:,half,1][mov_i[:,half,3,:]]
                    snr_wf_2d_01_v[:,1] += snr_wf_01[:,half,1][mov_i[:,half,1,:]]
                    snr_wf_2d_01_v[:,0] += snr_wf_01[:,half,2][mov_i[:,half,0,:]]
                    snr_wf_2d_01_v[:,4] += snr_wf_01[:,half,2][mov_i[:,half,4,:]]

                    snr_wf_2d_h[:,2] += snr_wf[:,half+8,0][mov_i[:,half+8,2,:]]
                    snr_wf_2d_h[:,3] += snr_wf[:,half+8,1][mov_i[:,half+8,3,:]]
                    snr_wf_2d_h[:,1] += snr_wf[:,half+8,1][mov_i[:,half+8,1,:]]
                    snr_wf_2d_h[:,0] += snr_wf[:,half+8,2][mov_i[:,half+8,0,:]]
                    snr_wf_2d_h[:,4] += snr_wf[:,half+8,2][mov_i[:,half+8,4,:]]

                    snr_wf_2d_01_h[:,2] += snr_wf_01[:,half+8,0][mov_i[:,half+8,2,:]]
                    snr_wf_2d_01_h[:,3] += snr_wf_01[:,half+8,1][mov_i[:,half+8,3,:]]
                    snr_wf_2d_01_h[:,1] += snr_wf_01[:,half+8,1][mov_i[:,half+8,1,:]]
                    snr_wf_2d_01_h[:,0] += snr_wf_01[:,half+8,2][mov_i[:,half+8,0,:]]
                    snr_wf_2d_01_h[:,4] += snr_wf_01[:,half+8,2][mov_i[:,half+8,4,:]]

                evt_snr.append(np.nanmax((snr_wf_2d_v+snr_wf_2d_h) / (snr_wf_2d_01_v+snr_wf_2d_01_h)))
                evt_snr_v.append(np.nanmax(snr_wf_2d_v / snr_wf_2d_01_v))
                evt_snr_h.append(np.nanmax(snr_wf_2d_h / snr_wf_2d_01_h))
                evt_num.append(event)
                trigger.append(trig_checker(rawEvt))

                if event == sel_evt:

                    #event-wise plot
                    from tools.plot import plot_16_overlap
                    from tools.plot import plot_1
                    from tools.plot import sky_map

                    snr_wf_2d_ex = np.full(mov_i.shape,np.nan)
        
                    for a in range(num_Ants):
                        snr_wf_2d_ex[:,a,2] = snr_wf[:,a,0][mov_i[:,a,2,:]]
                        snr_wf_2d_ex[:,a,3] = snr_wf[:,a,1][mov_i[:,a,3,:]]
                        snr_wf_2d_ex[:,a,1] = snr_wf[:,a,1][mov_i[:,a,1,:]]
                        snr_wf_2d_ex[:,a,0] = snr_wf[:,a,2][mov_i[:,a,0,:]]
                        snr_wf_2d_ex[:,a,4] = snr_wf[:,a,2][mov_i[:,a,4,:]]
    
                    snr_wf_2d_01_ex = np.copy(snr_wf_2d_ex)
                    snr_wf_2d_01_ex[snr_wf_2d_ex != 0] = 1
            
                    v_sum = np.nansum(snr_wf_2d_ex[:,:8], axis=1)
                    v_sum_01 = np.nansum(snr_wf_2d_01_ex[:,:8], axis=1)
                    h_sum = np.nansum(snr_wf_2d_ex[:,8:], axis=1)
                    h_sum_01 = np.nansum(snr_wf_2d_01_ex[:,8:], axis=1)
                    del snr_wf_2d_01_ex           
 
                    v_avg = v_sum / v_sum_01
                    h_avg = h_sum / h_sum_01
    
                    evt_snr_v_sky = np.nanmax(v_avg)
                    evt_snr_h_sky = np.nanmax(h_avg)
        
                    evt_snr_v_loc = np.where(v_avg == evt_snr_v_sky)
                    evt_snr_h_loc = np.where(h_avg == evt_snr_h_sky)
                    del evt_snr_v_sky, evt_snr_h_sky
            
                    evt_snr_v_sky_2d = np.nanmax(v_avg,axis=0)
                    evt_snr_h_sky_2d = np.nanmax(h_avg,axis=0)

                    v_match = snr_wf_2d_ex[:,:8,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
                    h_match = snr_wf_2d_ex[:,8:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
                    del snr_wf_2d_ex

                    v_sum_match = v_sum[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
                    h_sum_match = h_sum[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
                    del v_sum, h_sum

                    v_sum_match_01 = v_sum_01[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
                    h_sum_match_01 = h_sum_01[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
                    del v_sum_01, h_sum_01 

                    v_avg_match = v_avg[:,evt_snr_v_loc[1][0],evt_snr_v_loc[2][0]]
                    h_avg_match = h_avg[:,evt_snr_h_loc[1][0],evt_snr_h_loc[2][0]]
                    del v_avg, h_avg, evt_snr_v_loc, evt_snr_h_loc

                    plot_16_overlap(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', trig_type+' Roll Max SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Vpol'
                                    ,mov_t,v_match
                                    ,Output,trig_type+'_Roll_Max_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Vpol.png'
                                    ,'Vpol'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max SNR plot w/o ArrT Vpol was generated!')
                    
                    plot_16_overlap(r'Offset Time [ $ns$ ]',r'SNR [ $V/RMS$ ]', trig_type+' Roll Max SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Hpol'
                                    ,mov_t,h_match
                                    ,Output,trig_type+'_Roll_Max_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Hpol.png'
                                    ,'Hpol'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max SNR plot w/o ArrT Hpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Sum SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Vpol'
                                    ,mov_t, v_sum_match
                                    ,Output,trig_type+'_Roll_Max_Sum_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Vpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Sum SNR plot w/o ArrT Vpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Sum SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Hpol'
                                    ,mov_t, h_sum_match
                                    ,Output,trig_type+'_Roll_Max_Sum_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Hpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Sum SNR plot w/o ArrT Hpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Sum SNR(pw:'+str(peak_w)+'ns) Counts w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Vpol'
                                    ,mov_t, v_sum_match_01
                                    ,Output,trig_type+'_Roll_Max_Sum_SNR_Counts_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Vpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Sum SNR Counts plot w/o ArrT Vpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Sum SNR(pw:'+str(peak_w)+'ns) Counts w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Hpol'
                                    ,mov_t, h_sum_match_01
                                    ,Output,trig_type+'_Roll_Max_Sum_SNR_Counts_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Hpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Sum SNR Counts plot w/o ArrT Hpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Avg SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Vpol'
                                    ,mov_t, v_avg_match
                                    ,Output,trig_type+'_Roll_Max_Avg_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Vpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Avg SNR plot w/o ArrT Vpol was generated!')

                    plot_1(r'Offset Time [ $ns$ ]',r'Sum SNR [ $V/RMS$ ]', trig_type+' Roll Max Avg SNR(pw:'+str(peak_w)+'ns) w/o ArrT, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Hpol'
                                    ,mov_t, h_avg_match
                                    ,Output,trig_type+'_Roll_Max_Avg_SNR_pw'+str(peak_w)+'_wo_ArrT_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Hpol.png'
                                    ,'Event# '+str(event)+' '+trig_type+' Roll Max Avg SNR plot w/o ArrT Hpol was generated!')
          
                    nadir_range = np.arange(0+theta_w/2, 180, theta_w)
                    phi_range = np.arange(0+theta_w/2, 360, theta_w)
 
                    sky_map(evt_snr_v_sky_2d,nadir_range, phi_range, theta_w
                               , trig_type+', Evt-Wise SNR(pw:'+str(peak_w)+'ns) 2d sky, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Vpol'
                               ,Output, trig_type+'_Evt_SNR_pw'+str(peak_w)+'_2d_sky_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Vpol.png'
                               ,'Event# '+str(event)+' '+trig_type+' Event-wise SNR 2d sky map Vpol was generated!')

                    sky_map(evt_snr_h_sky_2d,nadir_range, phi_range, theta_w
                               , trig_type+', Evt-Wise SNR(pw:'+str(peak_w)+'ns) 2d sky, A'+str(Station)+', Run'+str(Run)+', Evt'+str(event)+', Hpol'
                               ,Output, trig_type+'_Evt_SNR_pw'+str(peak_w)+'_2d_sky_A'+str(Station)+'_Run'+str(Run)+'_Evt'+str(event)+'_Hpol.png'
                               ,'Event# '+str(event)+' '+trig_type+' Event-wise SNR 2d sky map Hpol was generated!')

                else:
                    pass

        del usefulEvent

    del mov_i, pad_t_l, p_len_front, p_len_end, ps_len_i, mov_t, pad_t, tale_i_front, tale_i_end, ant_arr_01, snr_wf, snr_wf_01, snr_wf_2d_v, snr_wf_2d_01_v, snr_wf_2d_h, snr_wf_2d_01_h, half_ant

    evt_snr = np.asarray(evt_snr)
    evt_snr_v = np.asarray(evt_snr_v)
    evt_snr_h = np.asarray(evt_snr_h)
    evt_num = np.asarray(evt_num)
    trigger = np.asarray(trigger)

    # hist map
    from tools.plot import hist_map
    hist_map(r'Averaged & event-wise SNRs [ $V/RMS$ ]',r'# of events', 'Evt-wise SNR Hist., A'+str(Station)+', Run'+str(Run)
                ,trigger,evt_snr_v,evt_snr_h
                ,Output,'Evt_SNR_hist_A'+str(Station)+'_Run'+str(Run)+'.png'
                ,'Event-wisc SNR histogram was generated!')

    print('Event-wise SNR making is done!')

    return evt_snr, evt_snr_v, evt_snr_h, evt_num, trigger, trig_index, ant_arr_copy, ant_arr_fft, ant_arr_fft_band, snr_wf_copy, snr_wf_copy_1, snr_wf_r_max_copy, v_match, h_match, v_sum_match, h_sum_match, v_sum_match_01, h_sum_match_01, v_avg_match, h_avg_match, evt_snr_v_sky_2d, evt_snr_h_sky_2d, nadir_range, phi_range, np.arange(time_pad_i, time_pad_f+t_width_ns, t_width_ns), mov_t, pad_t, peak_w
                 













