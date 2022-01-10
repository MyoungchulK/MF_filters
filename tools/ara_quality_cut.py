import os, sys
import numpy as np
import ROOT
from tqdm import tqdm
import h5py

# custom lib
from tools.ara_constant import ara_const

#link AraRoot
ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

ara_const = ara_const()
num_ddas = ara_const.DDA_PER_ATRI
num_blocks = ara_const.BLOCKS_PER_DDA
num_ants = ara_const.USEFUL_CHAN_PER_STATION

def quick_qual_check(dat_bool, dat_idx, ser_val):

    bool_len = np.count_nonzero(dat_bool)
    if bool_len > 0:
        print(f'Qcut, {ser_val}:', bool_len, dat_idx[dat_bool])
    del bool_len

class pre_qual_cut_loader:

    def __init__(self, ara_uproot, trim_1st_blk = False):

        self.st = ara_uproot.station_id
        self.run = ara_uproot.run
        self.trig_type = ara_uproot.get_trig_type()
        self.evt_num = ara_uproot.evt_num 
        self.unix_time = ara_uproot.unix_time
        self.irs_block_number = ara_uproot.irs_block_number
        self.pps_number = ara_uproot.pps_number
        self.remove_1_blk = int(trim_1st_blk)
        read_win = ara_uproot.read_win
        self.blk_len_arr = read_win//num_ddas - self.remove_1_blk

    def get_bad_event_number(self):
        
        bad_evt_num = np.full((len(self.evt_num)), 0, dtype = int)

        negative_idx = np.where(np.diff(self.evt_num) < 0)[0]
        if len(negative_idx) > 0:
            bad_evt_num[negative_idx[0] + 1:] = 1
        del negative_idx

        quick_qual_check(bad_evt_num != 0, self.evt_num, 'bad evt num')

        return bad_evt_num

    def get_bad_unix_time_events(self):

        known_issue = known_issue_loader(self.st)

        bad_unix_evts = np.full((len(self.unix_time)), 0, dtype = int)
        for evt in range(len(self.unix_time)):
            bad_unix_evts[evt] = known_issue.get_bad_unixtime(self.unix_time[evt])
        del known_issue

        quick_qual_check(bad_unix_evts != 0, self.evt_num, 'bad unix time')

        return bad_unix_evts
        
    def get_bad_readout_win_events(self, readout_limit = 26):

        if self.st == 2:
            if self.run < 4029:
                readout_limit = 20
            if self.run > 4028 and self.run < 9749:
                readout_limit = 26
            if self.run > 9748:
                readout_limit = 28
        if self.st == 3:
            if self.run < 3104:
                readout_limit = 20
            if self.run > 3103 and self.run < 10001:
                readout_limit = 26
            if self.run > 10000:
                readout_limit = 28
        readout_limit -= self.remove_1_blk

        bad_readout_win_evts = (self.blk_len_arr < readout_limit).astype(int) 
        bad_readout_win_evts[self.trig_type != 0] = 0

        quick_qual_check(bad_readout_win_evts != 0, self.evt_num, 'bad readout window events')

        return bad_readout_win_evts

    def get_zero_block_events(self, zero_blk_limit = 2):

        zero_blk_limit -= self.remove_1_blk

        zero_blk_evts = (self.blk_len_arr < zero_blk_limit).astype(int)

        quick_qual_check(zero_blk_evts != 0, self.evt_num, 'zero block')

        return zero_blk_evts

    def get_block_gap_events(self):

        blk_gap_evts = np.full((len(self.irs_block_number)), 0, dtype = int)

        for evt in range(len(self.irs_block_number)):
            irs_block_evt = self.irs_block_number[evt]
            first_block_idx = irs_block_evt[0]
            last_block_idx = irs_block_evt[-1]
            block_diff = len(irs_block_evt)//num_ddas - 1

            if first_block_idx + block_diff != last_block_idx:
                if num_blocks - first_block_idx + last_block_idx != block_diff:
                    blk_gap_evts[evt] = 1
            del irs_block_evt, first_block_idx, last_block_idx, block_diff

        quick_qual_check(blk_gap_evts != 0, self.evt_num, 'block gap')

        return blk_gap_evts

    def get_first_few_events(self, first_evt_limit = 7):

        first_few_evts = np.full((len(self.evt_num)), 0, dtype = int)

        if self.st == 2:
            first_few_evts[(self.evt_num < first_evt_limit) & (self.unix_time >= 1448485911)] = 1
        if self.st == 3:
            first_few_evts[self.evt_num < first_evt_limit] = 1

        quick_qual_check(first_few_evts != 0, self.evt_num, f'first few events')

        return first_few_evts
    
    def run_pre_qual_cut(self):

        tot_pre_qual_cut = np.full((len(self.evt_num), 6), 0, dtype = int)

        tot_pre_qual_cut[:,0] = self.get_bad_event_number()
        tot_pre_qual_cut[:,1] = self.get_bad_unix_time_events()
        tot_pre_qual_cut[:,2] = self.get_bad_readout_win_events()
        tot_pre_qual_cut[:,3] = self.get_zero_block_events()
        tot_pre_qual_cut[:,4] = self.get_block_gap_events()
        tot_pre_qual_cut[:,5] = self.get_first_few_events()
        # time stamp cut

        quick_qual_check(np.nansum(tot_pre_qual_cut, axis = 1) != 0, self.evt_num, 'total pre qual cut!')

        return tot_pre_qual_cut

class post_qual_cut_loader:

    def __init__(self, ara_uproot, ara_root, dt = 0.5):

        from tools.ara_wf_analyzer import wf_analyzer
        self.wf_int = wf_analyzer(dt = dt)
        self.dt = self.wf_int.dt
        self.evt_num = ara_uproot.evt_num
        self.ara_root = ara_root
        self.st_arr = np.arange(num_ddas, dtype = int)

        known_issue = known_issue_loader(ara_uproot.station_id)
        self.bad_ant = known_issue.get_bad_antenna(ara_uproot.run)
        del known_issue

        self.zero_adc_ratio = np.full((num_ants, len(self.evt_num)), np.nan, dtype = float)
        self.freq_glitch_evts = np.copy(self.zero_adc_ratio)
        self.spikey_evts = np.full((len(self.evt_num)), np.nan, dtype = float) 
        self.timing_err_evts = np.full((num_ants, len(self.evt_num)), 0, dtype = int)
        # spare
        # cliff (time stamp)?
        # overpower
        # band pass cut(offset block)??
        # cw (testbad, phase, anita)
        # cal, surface

    def get_timing_error_events(self, raw_t):

        timing_err_flag = int(np.any(np.diff(raw_t)<0))

        return timing_err_flag

    def get_zero_adc_events(self, raw_v, raw_len, zero_adc_limit = 8):

        zero_ratio = np.count_nonzero(raw_v < zero_adc_limit)/raw_len

        return zero_ratio

    def get_freq_glitch_events(self, raw_t, raw_v):

        int_v = self.wf_int.get_int_wf(raw_t, raw_v)[1]

        fft_peak_idx = np.nanargmax(np.abs(np.fft.rfft(int_v)))
        peak_freq = fft_peak_idx / (len(int_v) * self.dt)
        del int_v, fft_peak_idx

        return peak_freq

    def get_spikey_ratio(self, dat_ant, sel_st = 0, apply_bad_ant = False):

        if apply_bad_ant == True:
            dat_ant[self.bad_ant] = np.nan

        avg_st_snr = np.full((num_ddas), np.nan, dtype = float)
        for string in range(num_ddas):
            avg_st_snr[string] = np.nanmean(dat_ant[string::num_ddas])

        rest_st = np.in1d(self.st_arr, sel_st, invert = True)
        spikey_ratio = avg_st_snr[sel_st] / np.nanmean(avg_st_snr[rest_st])
        del avg_st_snr, rest_st
        
        return spikey_ratio

    def get_string_flag(self, dat_bool, st_limit = 2, apply_bad_ant = False):

        dat_int = dat_bool.astype(int)

        if apply_bad_ant == True:
            dat_int[self.bad_ant] = 0

        flagged_events = np.full(dat_int.shape, 0, dtype = int)
        for string in range(num_ddas):
            dat_int_sum = np.nansum(dat_int[string::num_ddas], axis = 0)
            flagged_events[string::num_ddas] = (dat_int_sum > st_limit).astype(int)
            del dat_int_sum
        del dat_int

        return flagged_events

    def get_post_qual_cut(self, evt):

        self.ara_root.get_entry(evt)

        self.ara_root.get_useful_evt(self.ara_root.cal_type.kOnlyGoodADC)
        for ant in range(num_ants):
            raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)
            raw_len = len(raw_t) 
            if raw_len == 0:
                del raw_t, raw_v, raw_len
                self.ara_root.del_TGraph()
                self.ara_root.del_usefulEvt()
                return
        
            self.zero_adc_ratio[ant, evt] = self.get_zero_adc_events(raw_v, raw_len)
            self.timing_err_evts[ant, evt] = self.get_timing_error_events(raw_t)
            del raw_t, raw_v, raw_len
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

        if np.nansum(self.timing_err_evts[:,evt]) > 0:
            return
        
        self.ara_root.get_useful_evt(self.ara_root.cal_type.kLatestCalib)
        v_peak = np.full((num_ants), np.nan, dtype = float)
        for ant in range(num_ants):
            raw_t, raw_v = self.ara_root.get_rf_ch_wf(ant)   
 
            self.freq_glitch_evts[ant, evt] = self.get_freq_glitch_events(raw_t, raw_v)
            v_peak[ant] = np.nanmax(np.abs(raw_v))
            del raw_t, raw_v
            self.ara_root.del_TGraph()
        self.ara_root.del_usefulEvt()

        self.spikey_evts[evt] = self.get_spikey_ratio(v_peak, apply_bad_ant = True) 
        del v_peak

    def run_post_qual_cut(self):

        tot_post_qual_cut = np.full((num_ants, len(self.evt_num), 5), 0, dtype = int)

        tot_post_qual_cut[self.bad_ant,:,0] = 1 # knwon bad antenna
        tot_post_qual_cut[:,:,1] = np.nansum(self.timing_err_evts, axis = 0)[np.newaxis, :]
        low_freq_limit = 0.13
        tot_post_qual_cut[:,:,2] = self.get_string_flag(self.freq_glitch_evts < low_freq_limit, apply_bad_ant = True)
        ratio_limit = 0
        tot_post_qual_cut[:,:,3] = self.get_string_flag(self.zero_adc_ratio > ratio_limit, apply_bad_ant = True) 
        spikey_limit = 100000
        tot_post_qual_cut[:,:,4] = (self.spikey_evts > spikey_limit).astype(int)[np.newaxis, :]

        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,0], axis = 0) != 0, self.evt_num, 'known bad antenna!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,1], axis = 0) != 0, self.evt_num, 'timing issue!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,2], axis = 0) != 0, self.evt_num, 'frequency glitch!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,3], axis = 0) != 0, self.evt_num, 'zero adc ratio!')
        quick_qual_check(np.nansum(tot_post_qual_cut[:,:,4], axis = 0) != 0, self.evt_num, 'spikey ratio!')
        quick_qual_check(np.nansum(tot_post_qual_cut, axis = (0,2)) != 0, self.evt_num, 'total post qual cut!')

        return tot_post_qual_cut

class qual_cut_loader:

    def __init__(self, ara_root, ara_uproot, dt = 0.5, trim_1st_blk = False):

        self.pre_qual = pre_qual_cut_loader(ara_uproot, trim_1st_blk = trim_1st_blk)
        self.post_qual = post_qual_cut_loader(ara_uproot, ara_root, dt = dt)

class clean_event_loader:

    def __init__(self, ara_uproot, trig_flag = None, qual_flag = None):

        print(f'Clean event type! Trig type: {trig_flag}, Qual type: {qual_flag}')

        self.st = ara_uproot.station_id
        self.run = ara_uproot.run

        self.evt_num = ara_uproot.evt_num
        self.entry_num = ara_uproot.entry_num
        self.trig_type = ara_uproot.get_trig_type()

        self.trig_flag = np.asarray(trig_flag)
        self.qual_flag = np.asarray(qual_flag)

    def get_clean_events(self, pre_cut, post_cut):

        tot_pre_cut = np.copy(pre_cut)
        if 2 in self.trig_flag:
            print('Untagged software WF filter is excluded!')
            tot_pre_cut[:, 2] = 0
        tot_pre_cut = np.nansum(tot_pre_cut, axis = 1)
        tot_post_cut = np.nansum(post_cut, axis = 2)

        trig_idx = np.in1d(self.trig_type, self.trig_flag)
        qual_idx = np.in1d(tot_pre_cut, self.qual_flag)
        tot_idx = (trig_idx & qual_idx)

        clean_evt = self.evt_num[tot_idx]
        clean_entry = self.entry_num[tot_idx]
        clean_ant = tot_post_cut[:, tot_idx]
        del trig_idx, qual_idx, tot_idx, tot_pre_cut, tot_post_cut

        print('total # of clean event:',len(clean_evt))

        return clean_evt, clean_entry, clean_ant

    def get_qual_cut_results(self):

        d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{self.st}/Qual_Cut/'
        d_path += f'Qual_Cut_A{self.st}_R{self.run}.h5'
        qual_file = h5py.File(d_path, 'r')
        print(f'{d_path} is loaded!')

        pre_qual_cut = qual_file['pre_qual_cut'][:]
        post_qual_cut = qual_file['post_qual_cut'][:]
            
        clean_evt, clean_entry, clean_ant = self.get_clean_events(pre_qual_cut, post_qual_cut)
        del d_path, qual_file, pre_qual_cut, post_qual_cut
        
        if len(clean_evt) == 0:
            print('There are no desired events!')
            sys.exit(1)

        return clean_evt, clean_entry, clean_ant

class known_issue_loader:

    def __init__(self, st):

        self.st = st
      
    def get_bad_antenna(self, run):

        # masked antenna
        bad_ant = np.array([], dtype = int)

        if self.st == 2:
            bad_ant = np.array([15], dtype = int) #D4BH

        if self.st ==  3:
            if run > 1901 and run < 10001:
                bad_ant = np.array([3,7,11,15], dtype = int)# all D4 antennas
            if run > 12865:
                bad_ant = np.array([0,4,8,12], dtype = int) # all D1 antennas, dead bit issue

        return bad_ant

    def get_good_antenna(self, run):

        bad_ant = self.get_bad_antenna(run)
        ant_idx = np.arange(num_ants)
        good_ant_bool = np.in1d(ant_idx, bad_ant, invert = True)
        good_ant = ant_idx[good_ant_bool]
        del bad_ant, ant_idx, good_ant_bool

        return good_ant

    def get_bad_unixtime(self, unix_time):

        # masked unixtime(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L503

        bad_unit_time = False

        if self.st == 2:

            # Livetime flagged as bad by Biran
            if((unix_time>=1389381600 and unix_time<=1389384000) or # from run 2868
            (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
            # (unix_time>=1449189600 and unix_time<=1449190200) or # from run 6507
            (unix_time>=1449187200 and unix_time<=1449196200) or # from run 6507

            #Livetime flagged as bad by Biran's undergrads
            #config 1
            # (unix_time>=1380234000 and unix_time<=1380236400) or # from run 2428 22 hour balloon launch
            # (unix_time>=1382046000 and unix_time<=1382047500) or # from run 2536 22 hour balloon launch
            (unix_time>=1382712900 and unix_time<=1382713500) or # from run 2575
            (unix_time>=1382972700 and unix_time<=1382973300) or # from run 2589
            # (unix_time>=1383689100 and unix_time<=1383690900) or # from run 2631 22 hour balloon launch
            (unix_time>=1383884400 and unix_time<=1383886200) or # from run 2642
            (unix_time>=1384060200 and unix_time<=1384061100) or # from run 2652
            (unix_time>=1384487400 and unix_time<=1384489800) or # from run 2677
            (unix_time>=1384489980 and unix_time<=1384491060) or # from run 2678 at start may be glitch or continued from 2677
            (unix_time>=1384856520 and unix_time<=1384856640) or # from run 2698 super zoomed in two minute window
            # (unix_time>=1385674200 and unix_time<=1385675100) or # from run 2744 22 hour balloon launch
            (unix_time>=1389381600 and unix_time<=1389383700) or # from run 2868 first of two from run 2868
            (unix_time>=1389398700 and unix_time<=1389400200) or # from run 2868 second of two from run 2868
            (unix_time>=1389665100 and unix_time<=1389666300) or # from run 2884
            (unix_time>=1393288800 and unix_time<=1393289400) or # from run 3099
            # (unix_time>=1397856600 and unix_time<=1397858400) or # from run 3442 22 hour balloon launch

            #config 2
            (unix_time>=1376731800 and unix_time<=1376733000) or # from run 2235

            #conifg 3
            (unix_time>=1400276700 and unix_time<=1400277300) or # from run 3605 mainly looks like glitch at end

            #config 4
            (unix_time>=1409986500 and unix_time<=1409988000) or # from run 4184
            # (unix_time>=1412026200 and unix_time<=1412027100) or # from run 4301 22 hr balloon
            # (unix_time>=1412285400 and unix_time<=1412288100) or # from run 4316 weird 22hr balloon
            # (unix_time>=1412544600 and unix_time<=1412545500) or # from run 4331 22hr balloon
            # (unix_time>=1412803800 and unix_time<=1412804700) or # from run 4346 22hr balloon
            (unix_time>=1413898200 and unix_time<=1413899100) or # from run 4408
            (unix_time>=1414083900 and unix_time<=1414086000) or # from run 4418
            (unix_time>=1414350300 and unix_time<=1414351200) or # from run 4434 pt 1
            # (unix_time>=1414358700 and unix_time<=1414359900) or # from run 4434 pt 2 22hr balloon
            (unix_time>=1414674300 and unix_time<=1414674780) or # from run 4452
            (unix_time>=1414986600 and unix_time<=1414987200) or # from run 4471
            (unix_time>=1415223000 and unix_time<=1415223900) or # from run 4483
            (unix_time>=1415380500 and unix_time<=1415381400) or # from run 4493
            (unix_time>=1415558100 and unix_time<=1415559000) or # from run 4503
            (unix_time>=1415742300 and unix_time<=1415743800) or # from run 4513
            (unix_time>=1416207000 and unix_time<=1416212100) or # from run 4541
            (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
            (unix_time>=1416905100 and unix_time<=1416910500) or # from run 4579 two spikes about an hour apart
            # (unix_time>=1416950700 and unix_time<=1416951600) or # from run 4582 22 hour balloon launch
            (unix_time>=1417677000 and unix_time<=1417678200) or # from run 4621  weird and cool
            (unix_time>=1417836000 and unix_time<=1417837500) or # from run 4631
            (unix_time>=1420097100 and unix_time<=1420098300) or # from run 4763
            (unix_time>=1420293300 and unix_time<=1420294200) or # from run 4774
            (unix_time>=1420317600 and unix_time<=1420318200) or # from run 4775
            (unix_time>=1420978200 and unix_time<=1420978800) or # from run 4814
            (unix_time>=1421024400 and unix_time<=1421025300) or # from run 4817
            (unix_time>=1421713200 and unix_time<=1421718600) or # from run 4872 looks full of errors and not spiky but could have a spiky
            (unix_time>=1421718000 and unix_time<=1421725800) or # from run 4873 definitely an error but also has spiky boy, part 1 of 2
            (unix_time>=1421733300 and unix_time<=1421733900) or # from run 4873 spiky boy alone but in a run with errors, part 2 of 2
            (unix_time>=1421783400 and unix_time<=1421794200) or # from run 4876 definitely an error but not a spikey boy
            # (unix_time>=1428529800 and unix_time<=1428530700) or # from run 5389 22 hour balloon launch
            (unix_time>=1435623000 and unix_time<=1435623600) or # from run 5801
            # (unix_time>=1436394000 and unix_time<=1436395200) or # from run 5845 22 hour balloon launch
            (unix_time>=1437601200 and unix_time<=1437602700) or # from run 5915 looks like error at the start
            # (unix_time>=1439933700 and unix_time<=1439934960) or # from run 6048 22 hour balloon launch
            (unix_time>=1440581700 and unix_time<=1440582480) or # from run 6086
            # (unix_time>=1441489200 and unix_time<=1441490280) or # from run 6137 22 hour balloon launch
            # (unix_time>=1444685400 and unix_time<=1444687080) or # from run 6322 22 hour balloon launch
            # (unix_time>=1445722020 and unix_time<=1445723220) or # from run 6383 22 hour balloon launch
            (unix_time>=1445934900 and unix_time<=1445935500) or # from run 6396
            (unix_time>=1445960400 and unix_time<=1445961000) or # from run 6397
            # (unix_time>=1445982120 and unix_time<=1445982900) or # from run 6398 22 hour balloon launch
            (unix_time>=1446165600 and unix_time<=1446166200) or # from run 6408
            # (unix_time>=1446327300 and unix_time<=1446328200) or # from run 6418 22 hour balloon launch
            (unix_time>=1446607800 and unix_time<=1446608640) or # from run 6433 looks like an error at end
            (unix_time>=1446784200 and unix_time<=1446784800) or # from run 6445
            # (unix_time>=1476739800 and unix_time<=1476741000) or # from run 8100 22 hour balloon launch
            # (unix_time>=1476999000 and unix_time<=1476999900) or # from run 8114 22 hour balloon launch but barely noticeable
            # (unix_time>=1477258200 and unix_time<=1477259100) or # from run 8129 22 hour balloon launch
            (unix_time>=1477511700 and unix_time<=1477512600) or # from run 8143 weird possible balloon launch
            (unix_time>=1477950300 and unix_time<=1477951500) or # from run 8168 22 hour balloon launch
            # (unix_time>=1478033400 and unix_time<=1478034000) or # from run 8173 22 hour balloon launch
            # (unix_time>=1478295300 and unix_time<=1478296200) or # from run 8188 22 hour balloon launch
            # (unix_time>=1478728500 and unix_time<=1478729400) or # from run 8213 22 hour balloon launch
            (unix_time>=1479231900 and unix_time<=1479232500) or # from run 8241

            # config 5
            (unix_time>=1449280500 and unix_time<=1449281100) or # from run 6513
            (unix_time>=1449610200 and unix_time<=1449612000) or # from run 6531
            (unix_time>=1450536000 and unix_time<=1450537200) or # from run 6584
            # (unix_time>=1450906200 and unix_time<=1450907100) or # from run 6606    22hr
            # (unix_time>=1451423700 and unix_time<=1451424600) or # from run 6635   22hr
            (unix_time>=1452008100 and unix_time<=1452009000) or # from run 6669
            # (unix_time>=1452115800 and unix_time<=1452116700) or # from run 6675    22hr
            (unix_time>=1452197700 and unix_time<=1452198600) or # from run 6679
            (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6680
            (unix_time>=1452282000 and unix_time<=1452282600) or # from run 6684
            (unix_time>=1452298200 and unix_time<=1452298800) or # from run 6685    possible error
            (unix_time>=1452385500 and unix_time<=1452386400) or # from run 6690
            # (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6694   22 hr
            (unix_time>=1452494100 and unix_time<=1452495000) or # from run 6696   possible error
            # (unix_time>=1452545100 and unix_time<=1452545880) or # from run 6700    could be error or 22hr
            # (unix_time>=1452636900 and unix_time<=1452637500) or # from run 6705   could be error or 22hr
            (unix_time>=1452715200 and unix_time<=1452716100) or # from run 6709   possible error
            (unix_time>=1452972300 and unix_time<=1452973440) or # from run 6724   possible error
            # (unix_time>=1453325400 and unix_time<=1453326600) or # from run 6743   22 hr
            (unix_time>=1453408500 and unix_time<=1453409400) or # from run 6747
            (unix_time>=1453930200 and unix_time<=1453931400) or # from run 6776
            # (unix_time>=1454535000 and unix_time<=1454536500) or # from run 6818   22 hr
            # (unix_time>=1455746400 and unix_time<=1455747900) or # from run 6889   22 hr
            (unix_time>=1456200900 and unix_time<=1456201800) or # from run 6916
            (unix_time>=1456392600 and unix_time<=1456393800) or # from run 6927
            (unix_time>=1456997400 and unix_time<=1456999200) or # from run 6962
            # (unix_time>=1457559000 and unix_time<=1457560800) or # from run 6994   22 hr
            (unix_time>=1460842800 and unix_time<=1460844600) or # from run 7119   22 hr // has CW contam cal pulsers
            # (unix_time>=1461620100 and unix_time<=1461621900) or # from run 7161   22 hr
            (unix_time>=1463002200 and unix_time<=1463004000) or # from run 7243  22 hr // has CW contam cal pulsers
            (unix_time>=1466501400 and unix_time<=1466503200) or # from run 7474
            (unix_time>=1466721900 and unix_time<=1466724600) or # from run 7486 22 hr // has CW contam cal pulsers
            (unix_time>=1466805600 and unix_time<=1466808300) or # from run 7489 22 hr // has CW contam cal pulsers
            (unix_time>=1466890200 and unix_time<=1466892000) or # from run 7494   22 hr // has CW contam cal pulsers
            (unix_time>=1467927600 and unix_time<=1467929700) or # from run 7552   22 hr
            # (unix_time>=1472333400 and unix_time<=1472335200) or # from run 7831   22 hr
            (unix_time>=1473111300 and unix_time<=1473112800) or # from run 7879    22 hr // has CW contam cal
            # (unix_time>=1473370500 and unix_time<=1473372900) or # from run 7899   22 hr
            # (unix_time>=1475011500 and unix_time<=1475013600) or # from run 7993   22 hr
            (unix_time>=1475185200 and unix_time<=1475187900) or # from run 8003 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1475358000 and unix_time<=1475359800) or # from run 8013 balloon 22h
            (unix_time>=1475529900 and unix_time<=1475531400) or # from run 8023 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1475702700 and unix_time<=1475704200) or # from run 8033 balloon 22hr
            (unix_time>=1476221400 and unix_time<=1476222300)): # from run 8069 balloon 22hr // has CW contam cal pulsers
            # (unix_time>=1476479700 and unix_time<=1476481800) # from run 8084 balloon 22hr

                bad_unit_time = True

        elif self.st == 3:

            # config 1 from undergrads
            if((unix_time>=1380234300 and unix_time<=1380235500) or # from run 1538, 22 hour balloon launch
            (unix_time>=1381008600 and unix_time<=1381010400) or # from run 1584, 22 hour balloon launch
            (unix_time>=1382476200 and unix_time<=1382477400) or # from run 1670, 22 hour balloon launch-ish
            (unix_time>=1382687400 and unix_time<=1382688600) or # from run 1682
            (unix_time>=1382712600 and unix_time<=1382713800) or # from run 1684, 15 hour spike
            (unix_time>=1382972700 and unix_time<=1382973300) or # from run 1698, 15 hour spike
            (unix_time>=1383688800 and unix_time<=1383691500) or # from run 1739, 22 hour balloon launch
            (unix_time>=1384060200 and unix_time<=1384060800) or # from run 1761
            (unix_time>=1384208700 and unix_time<=1384209900) or # from run 1770, 22 hour balloon launch
            (unix_time>=1384486200 and unix_time<=1384492800) or # from run 1786, repeated bursts over ~2 hrs
            (unix_time>=1389399600 and unix_time<=1389400800) or # from run 1980
            (unix_time>=1389744000 and unix_time<=1389747600) or # from run 2001, lots of activity, sweeps in phi
            (unix_time>=1390176600 and unix_time<=1390182000) or # from run 2025
            (unix_time>=1391027700 and unix_time<=1391028900) or # from run 2079, 22 hour balloon launch, but early?
            (unix_time>=1393652400 and unix_time<=1393660800) or # from run 2235, repeated bursts over ~2 hrs
            (unix_time>=1394846400 and unix_time<=1394856000) or # from run 2328, repeated bursts over ~2.5 hours
            (unix_time>=1395437400 and unix_time<=1395438600) or # from run 2363, 22 hour balloon launch
            (unix_time>=1397856300 and unix_time<=1397857800) or # from run 2526, 22 hour balloon launch

            # config 2
            (unix_time>=1390176600 and unix_time<=1390182000) or # from run 3533

            # config 3
            (unix_time>=1409954100 and unix_time<=1409956200) or # from run 3216, 22 hour balloon launch
            (unix_time>=1409986800 and unix_time<=1409988600) or # from run 3217
            (unix_time>=1412026200 and unix_time<=1412028000) or # from run 3332
            (unix_time>=1412284920 and unix_time<=1412287020) or # from run 3347, 22 hour balloon launch
            (unix_time>=1412544120 and unix_time<=1412546400) or # from run 3362, 22 hour balloon launch
            (unix_time>=1412803620 and unix_time<=1412805780) or # from run 3377, 22 hour balloon launch
            (unix_time>=1413897900 and unix_time<=1413899100) or # from run 3439
            (unix_time>=1413914400 and unix_time<=1413922200) or # from run 3440 big wide weird above ground
            (unix_time>=1414083600 and unix_time<=1414086300) or # from run 3449 , 2 spikes
            (unix_time>=1413550800 and unix_time<=1413552600) or # from run 3419, end of the run, before a software dominated run starts
            (unix_time>=1414674000 and unix_time<=1414675500) or # from run 3478
            (unix_time>=1415380500 and unix_time<=1415381400) or # from run 3520
            (unix_time>=1415460600 and unix_time<=1415461500) or # from run 3524
            (unix_time>=1415742000 and unix_time<=1415744100) or # from run 3540 22hr balloon
            (unix_time>=1416207300 and unix_time<=1416209700) or # from run 3568 2 small spikes
            (unix_time>=1416457800 and unix_time<=1416459000) or # from run 3579
            (unix_time>=1416909600 and unix_time<=1416910680) or # from run 3605
            (unix_time>=1416951000 and unix_time<=1416952500) or # from run 3608 22hr balloon
            (unix_time>=1417676400 and unix_time<=1417679400) or # from run 3647
            (unix_time>=1417742400 and unix_time<=1417743600) or # from run 3651
            (unix_time>=1417836600 and unix_time<=1417839300) or # from run 3656
            (unix_time>=1420317000 and unix_time<=1420318200) or # from run 3800
            (unix_time>=1420493700 and unix_time<=1420494600) or # from run 3810 22hr balloon
            (unix_time>=1420513200 and unix_time<=1420515000) or # from run 3811
            (unix_time>=1420598700 and unix_time<=1420600500) or # from run 3816
            (unix_time>=1420857900 and unix_time<=1420859700) or # from run 3830
            (unix_time>=1421019000 and unix_time<=1421020200) or # from run 3840 22hr balloon maybe?
            (unix_time>=1421101800 and unix_time<=1421103600) or # from run 3863 22hr balloon
            (unix_time>=1421723400 and unix_time<=1421723940) or # from run 3910
            (unix_time>=1421750700 and unix_time<=1421751720) or # from run 3912
            (unix_time>=1421868600 and unix_time<=1421881200) or # from run 3977 looks intentional
            (unix_time>=1421881200 and unix_time<=1421884680) or # from run 3978 continuation of thing above
            (unix_time>=1422048900 and unix_time<=1422049800) or # from run 3987 , 22 hour balloon launch
            (unix_time>=1422307200 and unix_time<=1422308100) or # from run 3995 22hr balloon
            (unix_time>=1423660800 and unix_time<=1423661700) or # from run 4132
            (unix_time>=1424819880 and unix_time<=1424820720) or # from run 4200
            (unix_time>=1428529500 and unix_time<=1428531000) or # from run 4412, 22 hour balloon launch
            (unix_time>=1429094400 and unix_time<=1429095600) or # from run 4445
            (unix_time>=1429615800 and unix_time<=1429617600) or # from run 4473
            (unix_time>=1429616700 and unix_time<=1429627500) or # from run 4474
            (unix_time>=1429733400 and unix_time<=1429734600) or # from run 4482
            (unix_time>=1431034500 and unix_time<=1431036900) or # from run 4557 , 22 hour balloon launch
            (unix_time>=1433365500 and unix_time<=1433367900) or # from run 4693
            (unix_time>=1435755600 and unix_time<=1435756500) or # from run 4829
            (unix_time>=1435791000 and unix_time<=1435791600) or # from run 4832
            (unix_time>=1436393700 and unix_time<=1436395500) or # from run 4867
            (unix_time>=1476740100 and unix_time<=1476741300) or # from run 7658
            (unix_time>=1477511400 and unix_time<=1477518300) or # from run 7704, big spike followed by nothing at all
            (unix_time>=1477604700 and unix_time<=1477605900) or # from run 7709,  22 hour balloon launch
            (unix_time>=1477950300 and unix_time<=1477951500) or # from run 7729
            (unix_time>=1479231600 and unix_time<=1479235800) or # from run 7802  , big spike followed by nothing at all

            # config 4
            (unix_time>=1448959200 and unix_time<=1448960100) or # from run 6009
            (unix_time>=1449610500 and unix_time<=1449611400) or # from run 6046 22 hour balloon launch
            (unix_time>=1450119900 and unix_time<=1450120500) or # from run 6077 possible 22 hour balloon launch
            (unix_time>=1450536360 and unix_time<=1450536720) or # from run 6098 spike is at end of time
            (unix_time>=1452116100 and unix_time<=1452116700) or # from run 6188 end of time and possible balloon launch
            (unix_time>=1452196800 and unix_time<=1452198600) or # from run 6193 could be balloon
            (unix_time>=1452213600 and unix_time<=1452214200) or # from run 6194
            (unix_time>=1452282300 and unix_time<=1452282900) or # from run 6198 could be balloon
            (unix_time>=1452298500 and unix_time<=1452299100) or # from run 6199 spike is at end of measured time
            (unix_time>=1452385800 and unix_time<=1452386400) or # from run 6203 spike is at end of measured time
            (unix_time>=1452457800 and unix_time<=1452458700) or # from run 6206 spike is at end of measured time, could be balloon
            (unix_time>=1452494100 and unix_time<=1452494700) or # from run 6208 spike is at end of measured time
            (unix_time>=1452544980 and unix_time<=1452545580) or # from run 6212 could be balloon
            (unix_time>=1452561120 and unix_time<=1452561480) or # from run 6213 spike is at end of measured time
            (unix_time>=1452637020 and unix_time<=1452637260) or # from run 6219 spike is at end of measured time, could be balloon
            (unix_time>=1452715320 and unix_time<=1452715680) or # from run 6223 spike is at end of measured time
            (unix_time>=1452972660 and unix_time<=1452973020) or # from run 6239 spike is at end of measured time
            (unix_time>=1453325400 and unix_time<=1453326300) or # from run 6259 could be balloon
            (unix_time>=1453930500 and unix_time<=1453931100) or # from run 6295 could be balloon
            (unix_time>=1454535000 and unix_time<=1454536200) or # from run 6328 could be balloon
            (unix_time>=1454911200 and unix_time<=1454911800) or # from run 6349 spike is at end of measured time could match below
            (unix_time>=1454911200 and unix_time<=1454912100) or # from run 6350 spike is at start of measured time could match above
            (unix_time>=1455746400 and unix_time<=1455747300) or # from run 6397 could be balloon
            (unix_time>=1456374300 and unix_time<=1456374900) or # from run 6433
            (unix_time>=1457559300 and unix_time<=1457560500) or # from run 6501 could be balloon
            (unix_time>=1460843100 and unix_time<=1460844600) or # from run 6618 spike is at start of measured time, could be balloon
            (unix_time>=1467927840 and unix_time<=1467929640) or # from run 7052 could be balloon
            (unix_time>=1473371280 and unix_time<=1473372180) or # from run 7458 could be balloon
            (unix_time>=1475186100 and unix_time<=1475187000) or # from run 7562 could be balloon
            (unix_time>=1475530500 and unix_time<=1475531700) or # from run 7584 could be balloon
            (unix_time>=1476221400 and unix_time<=1476222600)): # from run 7625 could be balloon

                bad_unit_time = True

        elif self.st == 5:
            pass

        return bad_unit_time

    def get_knwon_bad_run(self):

        bad_surface_run = self.get_bad_surface_run()
        bad_run = self.get_bad_run()
        knwon_bad_run = np.append(bad_surface_run, bad_run)
        special_run = self.get_L0_to_L1_Processing_Special_run()
        knwon_bad_run = np.append(knwon_bad_run, special_run)
        untagged_calpulser_run = self.get_untagged_calpulser_run()
        knwon_bad_run = np.append(knwon_bad_run, untagged_calpulser_run) 
        software_dominant_run = self.get_software_dominant_run()
        knwon_bad_run = np.append(knwon_bad_run, software_dominant_run)
        del bad_surface_run, bad_run, special_run, untagged_calpulser_run, software_dominant_run    
        print(f'Total number of known bad runs are {len(knwon_bad_run)}')

        return knwon_bad_run

    def get_L0_to_L1_Processing_Special_run(self):

        # from http://ara.icecube.wisc.edu/wiki/index.php/Data_processing_and_storage_plan
        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:
    
            # 2014
            #bad_run = np.append(bad_run, [2848, 2975, 2978, 2979, 3080, 3097, 3099]) # Half disk runs
            bad_run = np.append(bad_run, [3566, 2956, 2966, 2971, 4440, 2817, 2894, 2837, 2899, 
                                        2847, 2842, 2817, 4143, 2877, 2946, 2951, 2922, 2920, 
                                        2930, 2928, 2921, 2909, 2923, 2857, 2917, 2925, 2926, 
                                        2936, 2927, 2867, 2929, 2817, 2827, 2887, 2832, 2976, 2981]) # small runs
            #bad_run = np.append(bad_run, [2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819]) # Moved runs

            # 2015
            #bad_run = np.append(bad_run, [4820, 4821, 4822, 4823, 4825]) # Half disk runs
            bad_run = np.append(bad_run, [6166, 6141, 6042]) # small runs
            #bad_run = np.append(bad_run, [4762, 4763]) # Moved runs
            
            # 2016
            #bad_run = np.append(bad_run, []) # Half disk runs
            bad_run = np.append(bad_run, [7674, 7678, 7680, 7673, 7681, 7676, 7679]) # small runs
            #bad_run = np.append(bad_run, 6645) # Moved runs

            # 2017
            #bad_run = np.append(bad_run, [8530, 8575]) # Half disk runs
            """bad_run = np.append(bad_run, [8761, 8752, 8758, 8755, 8757, 8753, 8751, 8748, 8750, 
                                        8749, 8756, 8759, 8754, 8760, 1, 9044, 9049, 9047, 9050, 
                                        9052, 9048, 9053, 9042, 9043, 9046, 9045, 9041, 9051, 8656, 8657]) # small runs"""
            bad_run = np.append(bad_run, [8761, 8752, 8758, 8755, 8757, 8753, 8751, 8748, 8750,
                                        8749, 8756, 8759, 8754, 8760, 9044, 9049, 9047, 9050,
                                        9052, 9048, 9053, 9042, 9043, 9046, 9045, 9041, 9051, 8656, 8657]) # small runs
            #bad_run = np.append(bad_run, []) # Moved runs
            #bad_run = np.append(bad_run, 8530) # Duplicate runs

            # 2018
            #bad_run = np.append(bad_run, []) # Half disk runs
            bad_run = np.append(bad_run, [9964, 9968, 9965, 9966, 9967, 9984, 9987, 9986, 9985, 9983, 12338, 9560, 
                                        12040, 12042, 12043, 12041, 12044, 9769, 9778, 9834, 11632, 11630, 11628, 
                                        11637, 11638, 11631, 11633, 11635, 11627, 11639, 11640, 11641, 11634, 11636, 
                                        11629, 12528, 12529, 12436, 12465, 11274, 10435, 10440, 10441, 10434, 10439, 
                                        10438, 10437, 10436, 9562, 9508, 12445, 12444, 11138, 11139, 11142, 11145, 
                                        11140, 11144, 11141, 11143, 9772, 9795, 9777, 9779, 11256, 11252, 11240, 11249, 
                                        11260, 11261, 11273, 11262, 11271, 11253, 11237, 11272, 11250, 11239, 11259, 
                                        11235, 11246, 11242, 11238, 11258, 11265, 11267, 11247, 11255, 11263, 11241, 
                                        11264, 11243, 11234, 11257, 11233, 11268, 11245, 11236, 11251, 11248, 11270, 
                                        11266, 11269, 11254, 11244, 11136, 11129, 11137, 11135, 11134, 11133, 11130, 
                                        11132, 11128, 11131, 9747, 9783, 9746, 9782, 11458, 11415, 11453, 11454, 11456, 
                                        11410, 11439, 11451, 11428, 11429, 11452, 11449, 11434, 11412, 11417, 11441, 
                                        11421, 11444, 11423, 11418, 11427, 11422, 11425, 11413, 11438, 11414, 11440, 
                                        11442, 11420, 11446, 11432, 11457, 11430, 11426, 11455, 11416, 11459, 11448, 
                                        11419, 11437, 11436, 11424, 11450, 11409, 11431, 11435, 11445, 11433, 11447, 
                                        11411, 11443, 9776, 9774, 12586, 9781, 9770, 9824, 9833, 9835, 9836, 9842, 12315, 
                                        12303, 12319, 12302, 12294, 12298, 12300, 12321, 12311, 12296, 12313, 12310, 
                                        12297, 12306, 12309, 12295, 12320, 12308, 12317, 12301, 12305, 12318, 12304, 
                                        12307, 12316, 12299, 12530, 9768, 9784, 9841, 9773, 9811, 12394, 12324, 12322, 
                                        12323, 11027, 11028, 11041, 11032, 11035, 11024, 11022, 11023, 11034, 11029, 
                                        11025, 11036, 11042, 11031, 11026, 11039, 11033, 11040, 11020, 11030, 11038, 
                                        11037, 11021, 10794, 10796, 10797, 10795]) # small runs
            #bad_run = np.append(bad_run, []) # Moved runs
            #bad_run = np.append(bad_run, [12446, 11076, 9517, 9519, 9518, 9611, 9612]) # Duplicate runs
            #bad_run = np.append(bad_run, []) # Bad directory structure runs

        elif self.st == 3:

            # 2014
            #bad_run = np.append(bad_run, [2116, 2137, 2198]) # Half disk runs
            bad_run = np.append(bad_run, [2012, 2017, 1957, 1962, 1947, 2165, 2078, 2058, 2053, 2063, 1992, 1967, 
                                        1977, 1932, 2103, 1927, 1942, 1937, 2033, 2041, 2035, 2031, 2034, 2027]) # small runs
            #bad_run = np.append(bad_run, [1922, 1924, 1925, 1926, 1927, 929]) # Moved runs

            # 2015
            """bad_run = np.append(bad_run, [3945, 3943, 3921, 3933, 3947, 3928, 3932, 3934, 3936, 3944, 3966, 3919, 
                                        3942, 3972, 3953, 3957, 3961, 3927, 3971, 3974, 3925, 3962, 3951, 3965, 
                                        3968, 3950, 3963, 3931, 3952, 3941, 3940, 3955, 3967, 3946, 3852, 3859, 
                                        3849, 3858, 3847, 3844, 3851, 3854, 3848, 3855, 3860, 3845, 3853, 3850, 
                                        3846, 3857, 3856]) # Half disk runs"""
            #bad_run = np.append(bad_run, []) # small runs
            #bad_run = np.append(bad_run, [3784, 3785]) # Moved runs

            # 2016
            #bad_run = np.append(bad_run, []) # Half disk runs
            bad_run = np.append(bad_run, [6214, 6294, 6289, 6354, 6318, 6337, 6315, 6364, 6305, 6317, 6297, 6291]) # small runs
            #bad_run = np.append(bad_run, 6159) # Moved runs

            # 2018
            #bad_run = np.append(bad_run, []) # Half disk runs
            """bad_run = np.append(bad_run, [1304, 1288, 1304, 1288, 1300, 1302, 1658, 1654, 1662, 1652, 1688, 10066, 
                                        1689, 1809, 1774, 1796, 1798, 1785, 1766, 1668, 1771, 1650, 1694, 1814, 1700, 
                                        1686, 1770, 1763, 1791, 1690, 1804, 1767, 1786, 1783, 1680, 1795, 1776, 1775, 
                                        1768, 1808, 1773, 1788, 1790, 1806, 1779, 1801, 1717, 1799, 1781, 1789, 1764, 
                                        1811, 1685, 1769, 1671, 1687, 1765, 1805, 1800, 1663, 1675, 1793, 1803, 1780, 
                                        1784, 82643, 102, 103, 1778, 1684, 1706, 1794, 1810, 10169, 12884, 1360, 1664, 
                                        1661, 1304, 1288, 1300, 1302, 13040, 1347, 12759, 12925, 1352, 1660, 1345, 
                                        11933, 11934]) # small runs"""
            bad_run = np.append(bad_run, [10066,
                                        82643, 10169, 12884,
                                        13040, 12759, 12925,
                                        11933, 11934]) # small runs
            #bad_run = np.append(bad_run, []) # Moved runs
            #bad_run = np.append(bad_run, [1726, 1747, 12884, 11335, 1746, 10002, 10029, 10028, 10004, 10020]) # Duplicate runs
            #bad_run = np.append(bad_run, []) # Bad directory structure runs
            #bad_run = np.append(bad_run, [1643, 1646, 1726, 1727, 10035, 10042, 10043, 10051, 10052, 10053]) # Consolidated L1 pieces

        else:
            pass

        return bad_run

    def get_software_dominant_run(self):

        # https://github.com/clark2668/a23_analysis_tools/tree/master/data

        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:
            pass
        elif self.st == 3:
            bad_run = np.append(bad_run, [1795, 1796, 1797, 1799, 1800, 1801, 1802, 1804, 1805, 1806, 1807, 1809, 1810,
                                        1811, 1812, 1814, 1126, 1129, 1130, 1131, 1132, 1133, 1139, 1140, 1143, 1228, 1231,
                                        1322, 1428, 2000, 2037, 2038, 2042, 2043, 2466, 2467, 2468, 2469, 2471, 2472, 2473,
                                        3421, 3422, 3423, 3424, 3426, 3427, 3428, 3429, 3788, 3861, 3892, 3919, 4978, 5014,
                                        5024, 7756, 7757, 7758, 7760, 7761, 7762, 7763, 7765, 7766, 7767, 7768, 7770, 7771,
                                        7772, 7125, 7312, 7561, 7570])
        else:
            pass

        return bad_run

    def get_untagged_calpulser_run(self):

        # https://github.com/clark2668/a23_analysis_tools/tree/master/data
        
        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:
            pass
        elif self.st == 3:
            bad_run = np.append(bad_run, [1796, 1797,1799, 1800, 1801, 1802, 1804, 1805, 1806, 1807, 1809, 1810, 1811, 1812, 1814]) # c1
            bad_run = np.append(bad_run, [473, 476, 477, 478, 479, 480, 481, 484, 486, 487, 488, 489, 490, 491,
                                        492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
                                        507, 508, 801, 815, 892, 893, 894, 895, 896, 898, 899, 900, 901, 903, 904,
                                        905, 906, 908, 909, 910, 911, 913, 914, 915, 916, 918, 919, 920, 921, 923,
                                        935, 936, 937, 938, 939, 940, 942, 943, 944, 945, 947, 948, 949, 950, 952,
                                        953, 954, 955, 957, 958, 959, 960, 961, 962, 964, 965, 966, 967, 969, 970,
                                        971, 972, 974, 975, 976, 977, 979, 980, 981, 1125, 1126, 1128, 1129, 1130, 
                                        1131, 1132, 1133, 1134, 1135, 1136, 1138, 1141, 1142, 1143, 1144, 1145, 1146,   
                                        1148, 1149, 1150, 1151, 1153, 1154, 1155, 1156, 1158, 1159, 1160, 1161, 1163,
                                        1164, 1165, 1166, 1168, 1169, 1170, 1171, 1173, 1174, 1175, 1176, 1178, 1179, 
                                        1180, 1181, 1183, 1184, 1185, 1186, 1188, 1189, 1190, 1191, 1193, 1194, 1195, 
                                        1196, 1197, 1198, 1200, 1201, 1202, 1203, 1205, 1206, 1207, 1208, 1210, 1211, 
                                        1212, 1213, 1215, 1216, 1217, 1218, 1220, 1221, 1222, 1223, 1225, 1226, 1227, 
                                        1228, 1229, 1231, 1232, 1233, 1237, 1238, 1239, 1240, 1241, 1243, 1244, 1245, 
                                        1246, 1248, 1249, 1250, 1251, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 
                                        1263, 1264, 1265, 1267, 1268, 1269, 1270, 1272, 1273, 1274, 1275, 1277, 1278,
                                        1279, 1280, 1282, 1283, 1284, 1285, 1287, 1288, 1289, 1290, 1292, 1293, 1294,
                                        1295, 1308, 1312, 1313, 1322, 1330, 1331, 1332, 1333, 1334, 1335, 1337, 1338, 
                                        1339, 1340, 1341, 1342, 1344, 1345, 1346, 1347, 1349, 1350, 1351, 1352, 1354, 
                                        1355, 1356, 1357, 1359, 1360, 1361, 1362, 1364, 1365, 1366, 1367, 1369, 1370, 
                                        1371, 1372, 1374, 1375, 1376, 1377, 1379, 1380, 1381, 1382, 1384, 1385, 1387, 
                                        1389, 1390, 1391, 1392, 1394, 1395, 1396, 1397, 1399, 1400, 1401, 1402, 1404, 
                                        1405, 1406, 1407, 1413, 1414, 1416, 1417, 1418, 1419, 1421, 1422, 1423, 1424,
                                        1426, 1427, 1428]) # c2
            bad_run = np.append(bad_run, [3421, 3422, 3423, 3424, 3426, 3427, 3428, 3517, 3788, 3844, 3845, 3846, 3847, 
                                        3848, 3849, 3850, 3851, 3852, 3853, 3854, 3857, 3858, 3859, 3860, 3881, 3882, 
                                        3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3916, 3917, 3918, 3919, 
                                        3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932,
                                        3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 
                                        3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 
                                        3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 
                                        3972, 3973, 3974, 3975, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 
                                        4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 
                                        4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 
                                        4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 
                                        4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 
                                        4070, 4071, 4072, 4914, 4916, 4917, 4918, 4919, 4921, 4922, 4923, 4924, 4926,
                                        4927, 4928, 4929, 4931, 4932, 4933, 4934, 4936, 4937, 4938, 4939, 4941, 4942,
                                        4943, 4944, 4946, 4947, 4948, 4949, 4951, 4952, 4953, 4954, 4956, 4957, 4958, 
                                        4959, 7757, 7758, 7760, 7761, 7762, 7763, 7765, 7766, 7767, 7768, 7770, 7771,
                                        7772]) # c3
            bad_run = np.append(bad_run, [6509, 7146, 7147, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7157,
                                        7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 
                                        7185, 7186, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198,
                                        7199, 7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212,
                                        7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 
                                        7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240,
                                        7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253]) # c4
            bad_run = np.append(bad_run, [2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2388,
                                        2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2467, 2468,
                                        2469, 2471,2472]) # c5

            # new list
            bad_run = np.append(bad_run, [10001, 10002, 10003, 10006, 10007, 10008, 10009, 10010, 10012, 10013, 10014, 10015,
                                         10017, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029,
                                         10030, 10032, 10034, 10035, 10036, 10040, 10041, 10042, 10043, 10044, 10045, 10046,
                                         10047, 10048, 10049, 10050, 10051, 10052, 10053, 10055, 10056, 10057, 10058, 10059,
                                         10060, 10061, 10062, 10063, 10064, 10065, 10067, 10075, 10076, 10077, 10081, 10083,
                                         10167, 10168, 10513, 10590, 10591, 10592, 10593, 10594, 10659, 10660, 10661, 10662,
                                         10663, 10664, 10665, 10666, 10682, 10683, 10684, 10716, 10717, 10718, 10719, 10720,
                                         10875, 10876, 10877, 10878, 10879, 10880, 10882, 10883, 10884, 10885, 10886, 10887,
                                         10888, 10889, 10891, 10892, 10893, 10894, 10895, 11085, 11086, 11088, 11089, 11090,
                                         11091, 11092, 11093, 11094, 11123, 11325, 11326, 11650, 11651, 11652, 11653, 11723,
                                         11724, 11725, 11726, 11728, 11729, 11755, 11756, 11757, 11758, 11759, 11760, 11761,
                                         11762, 11825, 11826, 11827, 11828, 11829, 11831, 12105, 12106, 12107, 12108, 12109,
                                         12111, 12294, 12295, 12296, 12297, 12298, 12456, 12457, 12458, 12686, 12687, 12688,
                                         12689, 12690, 12795, 12797, 12799, 12800, 12801, 12802, 12803, 12804, 12805, 12806,
                                         12807, 12809, 12810, 12811, 12812, 12813, 12814, 12815, 12816, 12817, 12818, 12819,
                                         12820, 12821, 12822, 12823, 12824, 12825, 12826, 12827, 12828, 12829, 12830, 12866,
                                         12870])

        else:
            pass

        return bad_run

    def get_bad_surface_run(self):

        # masked run(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L782
        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:

            # Runs shared with Ming-Yuan
            # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889

            bad_run = np.append(bad_run, 2090)
            bad_run = np.append(bad_run, 2678)
            bad_run = np.append(bad_run, 4777)
            bad_run = np.append(bad_run, 5516)
            bad_run = np.append(bad_run, 5619)
            bad_run = np.append(bad_run, 5649)
            bad_run = np.append(bad_run, 5664)
            bad_run = np.append(bad_run, 5666)
            bad_run = np.append(bad_run, 5670)
            bad_run = np.append(bad_run, 5680)
            bad_run = np.append(bad_run, 6445)
            bad_run = np.append(bad_run, 6536)
            bad_run = np.append(bad_run, 6542)
            bad_run = np.append(bad_run, 6635)
            bad_run = np.append(bad_run, 6655)
            bad_run = np.append(bad_run, 6669)
            bad_run = np.append(bad_run, 6733)

            # Runs identified independently

            bad_run = np.append(bad_run, 2091)
            bad_run = np.append(bad_run, 2155)
            bad_run = np.append(bad_run, 2636)
            bad_run = np.append(bad_run, 2662)
            bad_run = np.append(bad_run, 2784)
            bad_run = np.append(bad_run, 4837)
            bad_run = np.append(bad_run, 4842)
            bad_run = np.append(bad_run, 5675)
            bad_run = np.append(bad_run, 5702)
            bad_run = np.append(bad_run, 6554)
            bad_run = np.append(bad_run, 6818)
            bad_run = np.append(bad_run, 6705)
            bad_run = np.append(bad_run, 8074)

        elif self.st == 3:

            # Runs shared with Ming-Yuan
            # http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=2041

            bad_run = np.append(bad_run, 977)
            bad_run = np.append(bad_run, 1240)
            bad_run = np.append(bad_run, 3158)
            bad_run = np.append(bad_run, 3431)
            bad_run = np.append(bad_run, 3432)
            bad_run = np.append(bad_run, 3435)
            bad_run = np.append(bad_run, 3437)
            bad_run = np.append(bad_run, 3438)
            bad_run = np.append(bad_run, 3439)
            bad_run = np.append(bad_run, 3440)
            bad_run = np.append(bad_run, 3651)
            bad_run = np.append(bad_run, 3841)
            bad_run = np.append(bad_run, 4472)
            bad_run = np.append(bad_run, 4963)
            bad_run = np.append(bad_run, 4988)
            bad_run = np.append(bad_run, 4989)

            # Runs identified independently

            bad_run = np.append(bad_run, 1745)
            bad_run = np.append(bad_run, 3157)
            bad_run = np.append(bad_run, 3652)
            bad_run = np.append(bad_run, 3800)
            bad_run = np.append(bad_run, 6193)
            bad_run = np.append(bad_run, 6319)
            bad_run = np.append(bad_run, 6426)

            # Runs I am sure we will exclude...

            bad_run = np.append(bad_run, 2000)
            bad_run = np.append(bad_run, 2001)

        else:
            pass

        return bad_run

    def get_bad_run(self):

        # masked run(2014~2016) from brian's analysis
        # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L881

        # array for bad run
        bad_run = np.array([], dtype=int)

        if self.st == 2:

            ## 2013 ##

            ## 2014 ##
            # 2014 rooftop pulsing, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, [3120, 3242])

            # 2014 surface pulsing
            # originally flagged by 2884, 2895, 2903, 2912, 2916
            # going to throw all runs jan 14-20
            bad_run = np.append(bad_run, 2884) # jan 14 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, [2885, 2889, 2890, 2891, 2893]) # exclusion by proximity

            bad_run = np.append(bad_run, 2895) # jan 16 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2898) # exclusion by proximity
            bad_run = np.append(bad_run, [2900, 2901, 2902]) # jan 17 2014. exclusion by proximity

            bad_run = np.append(bad_run, 2903) # # jan 18 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, [2905, 2906, 2907]) # exclusion by proximity

            bad_run = np.append(bad_run, 2912) # # jan 19 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2915) # exclusion by proximity

            bad_run = np.append(bad_run, 2916) # jan 20 2014 surface pulser runs. actual problem causer
            bad_run = np.append(bad_run, 2918) # exclusion by proximity

            # surface pulsing from m richman (identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14)
            bad_run = np.append(bad_run, [2938, 2939])

            # 2014 Cal pulser sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(3139, 3162+1))
            bad_run = np.append(bad_run, np.arange(3164, 3187+1))
            bad_run = np.append(bad_run, np.arange(3289, 3312+1))

            """
            # ARA02 stopped sending data to radproc. Alert emails sent by radproc.
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # http://ara.icecube.wisc.edu/wiki/index.php/Drop_29_3_2014_ara02
            bad_run = np.append(bad_run, 3336)
            """

            # 2014 L2 Scaler Masking Issue.
            # Cal pulsers sysemtatically do not reconstruct correctly, rate is only 1 Hz
            # Excluded because configuration was not "science good"
            bad_run = np.append(bad_run, np.arange(3464, 3504+1))

            # 2014 Trigger Length Window Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(3578, 3598+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # 2014, 4th June, Checking the functionality of the L1Scaler mask.
            bad_run = np.append(bad_run, 3695) # Masiking Ch0,1, 14
            bad_run = np.append(bad_run, 3700) # Masiking Ch2, 14
            bad_run = np.append(bad_run, 3701) # Masiking Ch4,5, 14
            bad_run = np.append(bad_run, 3702) # Masiking Ch6,7, 14
            bad_run = np.append(bad_run, 3703) # Masiking Ch8,9, 14
            bad_run = np.append(bad_run, 3704) # Masiking Ch10,11, 14
            bad_run = np.append(bad_run, 3705) # Masiking Ch12,13, 14
            bad_run = np.append(bad_run, 3706) # Masiking Ch14, 15

            # 2014, 16th June, Software update on ARA02 to fix the L1triggers.
            bad_run = np.append(bad_run, 3768)

            # 2014, 31st July, Testing new software to change trigger and readout window, pre-trigger samples.
            bad_run = np.append(bad_run, np.arange(3988, 3994+1))

            # 2014, 5th Aug, More tests on the pre-trigger samples.
            bad_run = np.append(bad_run, np.arange(4019, 4022+1))

            # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
            bad_run = np.append(bad_run, 4029)

            # 2014, 14th Aug, Finally changed trigger window size to 170ns.
            # http://ara.icecube.wisc.edu/wiki/index.php/File:Gmail_-_-Ara-c-_ARA_Operations_Meeting_Tomorrow_at_0900_CDT.pdf
            bad_run = np.append(bad_run, 4069)
            """
            
            ## 2015 ##
            # ??
            bad_run = np.append(bad_run, 4004)

            # 2015 icecube deep pulsing
            # 4787 is the "planned" run
            # 4795,4797-4800 were accidental
            bad_run = np.append(bad_run, 4785) # accidental deep pulser run (http://ara.physics.wisc.edu/docs/0017/001719/003/181001_ARA02AnalysisUpdate.pdf, slide 38)
            bad_run = np.append(bad_run, 4787) # deep pulser run (http://ara.physics.wisc.edu/docs/0017/001724/004/181015_ARA02AnalysisUpdate.pdf, slide 29)
            bad_run = np.append(bad_run, np.arange(4795, 4800+1))

            # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
            bad_run = np.append(bad_run, np.arange(4820, 4825+1))
            bad_run = np.append(bad_run, np.arange(4850, 4854+1))
            bad_run = np.append(bad_run, np.arange(4879, 4936+1))
            bad_run = np.append(bad_run, np.arange(5210, 5277+1))

            # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
            bad_run = np.append(bad_run, [4872, 4873])
            bad_run = np.append(bad_run, 4876) # Identified by MYL http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1889 slide 14

            # 2015 Pulser Lift, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 2)
            # Run number from private communication with John Kelley
            bad_run = np.append(bad_run, 6513)

            # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
            bad_run = np.append(bad_run, 6527)

            ## 2016 ##
            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016, 21st July, Reduced trigger delay by 100ns.
            bad_run = np.append(bad_run, 7623)
            """

            # 2016 cal pulser sweep, Jan 2015?, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            bad_run = np.append(bad_run, np.arange(7625, 7686+1))

            ## other ##
            # D1 Glitches, Identified by MYL as having glitches after long periods of downtime
            bad_run = np.append(bad_run, 3)
            bad_run = np.append(bad_run, 11)
            bad_run = np.append(bad_run, 59)
            bad_run = np.append(bad_run, 60)
            bad_run = np.append(bad_run, 71)

            # Badly misreconstructing runs
            # run 8100. Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015.
            bad_run = np.append(bad_run, np.arange(8100, 8246+1))

            ## 2017 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2017
            # 01/16/2017, Rooftop pulser run, Hpol ran for 30 min at 1 Hz starting 22:13:06. Vpol ran for 30 min at 1 Hz starting 22:44:50.
            bad_run = np.append(bad_run, 8530)

            # 01/24/2017, Deep pulser run, IC string 1 shallow pulser ~23:48-00:00. IC string 22 shallow pulser (Jan 25) ~00:01-00:19.
            bad_run = np.append(bad_run, 8573)

            # 01/25/2017, A2D6 pulser lift, Ran in continuous noise mode with V&Hpol Tx.
            bad_run = np.append(bad_run, [8574, 8575])

            # 01/25/2017, Same configuration as run8575, Ran in continuous noise mode with Hpol Tx. Forgot to switch back to normal configuration. No pulser lift in this period.
            bad_run = np.append(bad_run, [8576, 8578])

            # Cal pulser attenuation sweep
            """
            bad_run = np.append(bad_run, 8953) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            bad_run = np.append(bad_run, np.arange(8955, 8956+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            bad_run = np.append(bad_run, np.arange(8958, 8962+1)) # 04/10/2017, Data runs incorrectly tagged as calibration (this is real data!)
            """
            bad_run = np.append(bad_run, np.arange(8963, 9053+1)) # 04/10/2017, System crashed on D5 (D6 completed successfully); D6 VPol 0 dB is 8963...D6 VPol 31 dB is 8974...D6 HPol 0 dB is 8975...D5 VPol 0 dB is 9007...crashed before D5 HPol
            bad_run = np.append(bad_run, np.arange(9129, 9160+1)) # 04/25/2017, D6 VPol: 9129 is 0 dB, 9130 is 1 dB, ... , 9160 is 31 dB
            bad_run = np.append(bad_run, np.arange(9185, 9216+1)) # 05/01/2017, D6 HPol: 9185 is 0 dB, 9186 is 1 dB, ... , 9216 is 31 dB
            bad_run = np.append(bad_run, np.arange(9231, 9262+1)) # 05/04/2017, D5 VPol: 9231 is 0 dB, ... , 9262 is 31 dB
            bad_run = np.append(bad_run, np.arange(9267, 9298+1)) # 05/05/2017, D5 HPol: 9267 is 0 dB, ... , 9298 is 31 dB

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

            ## 2019 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2019
            # D5 Calpulser sweep, 01/25/2019
            bad_run = np.append(bad_run, np.arange(12842, 12873+1)) # D5 Vpol attenuation sweep 0 to 31 dB with a step of 1 dB.
            bad_run = np.append(bad_run, np.arange(12874, 12905+1)) # D5 Hpol attenuation sweep 0 to 31 dB with a step of 1 dB. Wanted to verify if D5 Hpol actually fires or not. Conclusion was that D5 Hpol does not fire and ARA02 defaults to firing D5 Vpol instead.

            # D6 Vpol fired at 0 dB attenuation. Trigger delays of ARA2 ch. adjusted.
            # 03/22/2019 ~ 04/11/2019
            bad_run = np.append(bad_run, np.arange(13449, 13454+1))
            bad_run = np.append(bad_run, np.arange(13455, 13460+1))
            bad_run = np.append(bad_run, np.arange(13516, 13521+1))
            bad_run = np.append(bad_run, np.arange(13522, 13527+1))
            bad_run = np.append(bad_run, np.arange(13528, 13533+1))
            bad_run = np.append(bad_run, 13542)
            bad_run = np.append(bad_run, np.arange(13543, 13547+1))
            bad_run = np.append(bad_run, 13549)
            bad_run = np.append(bad_run, np.arange(13550, 13554+1))
            bad_run = np.append(bad_run, np.arange(13591, 13600+1))
            bad_run = np.append(bad_run, np.arange(13614, 13628+1))
            bad_run = np.append(bad_run, np.arange(13630, 13644+1))
            bad_run = np.append(bad_run, np.arange(13654, 13663+1))
            bad_run = np.append(bad_run, np.arange(13708, 13723+1))
            bad_run = np.append(bad_run, np.arange(13732, 13746+1))
            bad_run = np.append(bad_run, np.arange(13757, 13771+1))
            bad_run = np.append(bad_run, np.arange(13772, 13775+1))

            # Trigger delays of ARA2 ch.
            # 04/18/2019 ~ 05/2/2019
            bad_run = np.append(bad_run, np.arange(13850, 13875+1))
            bad_run = np.append(bad_run, np.arange(13897, 13898+1))
            bad_run = np.append(bad_run, np.arange(13900, 13927+1))
            bad_run = np.append(bad_run, np.arange(13967, 13968+1))
            bad_run = np.append(bad_run, np.arange(13970, 13980+1))
            bad_run = np.append(bad_run, np.arange(13990, 14004+1))
            bad_run = np.append(bad_run, np.arange(14013, 14038+1))
            bad_run = np.append(bad_run, np.arange(14049, 14053+1))
            bad_run = np.append(bad_run, np.arange(14055, 14060+1))
            bad_run = np.append(bad_run, np.arange(14079, 14087+1))
            bad_run = np.append(bad_run, np.arange(14097, 14105+1))
            bad_run = np.append(bad_run, np.arange(14115, 14123+1))
            bad_run = np.append(bad_run, np.arange(14133, 14141+1))
            bad_run = np.append(bad_run, np.arange(14160, 14185+1))
            bad_run = np.append(bad_run, np.arange(14194, 14219+1))
            bad_run = np.append(bad_run, np.arange(14229, 14237+1))

            # need more investigation
            #bad_run = np.append(bad_run, 4829)
            #bad_run = np.append(bad_run, [8562, 8563, 8567, 8568, 8572])
            #bad_run = np.append(bad_run, 8577)
            #bad_run = np.append(bad_run, [9748, 9750])
            #bad_run = np.append(bad_run, np.arange(9522, 9849))

            # short run
            #bad_run = np.append(bad_run, 6480)
            #bad_run = np.append(bad_run, 10125)

        elif self.st == 3:

            ## 2013 ##
            # Misc tests: http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2013
            # bad_run = np.append(bad_run, np.arange(22, 62+1))

            # ICL rooftop: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # bad_run = np.append(bad_run, np.arange(63, 70+1))
            # bad_run = np.append(bad_run, np.arange(333, 341+1))

            # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # bad_run = np.append(bad_run, np.arange(72, 297+1))
            # bad_run = np.append(bad_run, np.arange(346, 473+1))

            # Eliminate all early data taking (all runs before 508)
            bad_run = np.append(bad_run, np.arange(508+1))

            # Cal sweep: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            # ??

            ## 2014 ##
            # 2014 Rooftop Pulser, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, [2235, 2328])

            # 2014 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(2251, 2274+1))
            bad_run = np.append(bad_run, np.arange(2376, 2399+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            # 2014, 6th Aug, Switched to new readout window: 25 blocks, pre-trigger: 14 blocks.
            bad_run = np.append(bad_run, 3063)

            # 2014, 14th Aug, Finally changed trigger window size to 170ns.
            bad_run = np.append(bad_run, 3103)
            """ 
            
            ## 2015 ##
            # 2015 surface or deep pulsing
            # got through cuts
            # happened jan 5-6, some jan 8
            # waveforms clearly show double pulses or things consistent with surface pulsing
            bad_run = np.append(bad_run, 3811)
            bad_run = np.append(bad_run, [3810, 3820, 3821, 3822]) # elminated by proximity to deep pulser run
            bad_run = np.append(bad_run, 3823) # deep pulser, observation of 10% iterator event numbers 496, 518, 674, 985, 1729, 2411

            # 2015 noise source tests, Jan, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2015
            bad_run = np.append(bad_run, np.arange(3844, 3860+1))
            bad_run = np.append(bad_run, np.arange(3881, 3891+1))
            bad_run = np.append(bad_run, np.arange(3916, 3918+1))
            bad_run = np.append(bad_run, np.arange(3920, 3975+1))
            bad_run = np.append(bad_run, np.arange(4009, 4073+1))

            # 2015 surface pulsing, Jan, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1339 (slide 5)
            bad_run = np.append(bad_run, [3977, 3978])

            # 2015 ICL pulsing, Dec, http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1269 (page 7)
            bad_run = np.append(bad_run, 6041)

            # 2015 station anomaly
            # see moni report: http://ara.physics.wisc.edu/cgi-bin/DocDB/ShowDocument?docid=1213
            # identified by MYL: http://ara.icecube.wisc.edu/wiki/index.php/A23_Diffuse_UW
            bad_run = np.append(bad_run, np.arange(4914, 4960+1))

            ## 2016 ##

            """
            http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016, 21st July, Reduced trigger delay by 100ns.
            bad_run = np.append(bad_run, 7124)
            """

            # More events with no RF/deep triggers, seems to precede coming test
            bad_run = np.append(bad_run, 7125)

            # 2016 Cal Pulser Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
            bad_run = np.append(bad_run, np.arange(7126, 7253+1))

            """
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2016
            # 2016 Loaded new firmware which contains the individual trigger delays which were lost since PCIE update in 12/2015.
            bad_run = np.append(bad_run, 7658)
            """

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx

            """
            # need more investigation
            bad_run = np.append(bad_run, np.arange(12788, 12832))
            bad_run = np.append(bad_run, np.arange(12866, 13087))

            # short run
            bad_run = np.append(bad_run, 1125)
            bad_run = np.append(bad_run, 1126)
            bad_run = np.append(bad_run, 1129)
            bad_run = np.append(bad_run, 1130)
            bad_run = np.append(bad_run, 1132)
            bad_run = np.append(bad_run, 1133)
            bad_run = np.append(bad_run, 1139)
            bad_run = np.append(bad_run, 1140)
            bad_run = np.append(bad_run, 1141)
            bad_run = np.append(bad_run, 1143)
            bad_run = np.append(bad_run, 10025)
            bad_run = np.append(bad_run, 10055)
            bad_run = np.append(bad_run, 11333)
            bad_run = np.append(bad_run, 11418)
            bad_run = np.append(bad_run, 11419)
            bad_run = np.append(bad_run, 12252)
            bad_run = np.append(bad_run, 12681)
            bad_run = np.append(bad_run, 12738)
            """

        elif self.st == 5:

            ## 2018 ##
            # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
            # Calibration pulser lowered, http://ara.physics.wisc.edu/docs/0015/001589/002/ARA5CalPulser-drop-Jan-2018.xlsx

            # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx
            bad_run = np.append(bad_run)

        else:
            pass

        return bad_run

