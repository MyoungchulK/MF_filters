import numpy as np

# information that can call directly data through AraRoot... later.....

def antenna_info():

    ant_name =['D1TV', 'D2TV', 'D3TV','D4TV','D1BV', 'D2BV', 'D3BV','D4BV','D1TH', 'D2TH', 'D3TH','D4TH','D1BH', 'D2BH', 'D3BH','D4BH']
    ant_index = np.arange(0,len(ant_name))
    num_ant = len(ant_name)

    return ant_name, ant_index, num_ant

def pol_type(pol_t = 2):

    return pol_t

def bad_antenna(st,Run):
    
    # masked antenna
    if st==2:
        #if Run>120 and Run<4028:
        #if Run>4027:
        bad_ant = np.array([15]) #D4BH
    
    elif st==3:
        #if Run<3014:
        #if Run>3013:
    
        if Run>1901:
            bad_ant = np.array([3,7,11,15])# all D4 antennas
        else:
            bad_ant = np.array([])

    else:
        bad_ant = np.array([])
    
    return bad_ant

def bad_run(st):

    # masked run(2014~2016) from brian's analysis 
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L881
    
    # array for bad run
    bad_run = np.array([], dtype=int)

    if st == 2:

        ## 2013 ##

        ## 2014 ##
        # 2014 rooftop pulsing, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.append(bad_run, [3120, 3242])

        # 2014 surface pulsing
        # originally flagged by 2884, 2895, 2903, 2912, 2916
        # going to throw all runs jan 14-20
        bad_run = np.append(ban_run, 2884) # jan 14 2014 surface pulser runs. actual problem causer
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
        bad_run = np.apprnd(bad_run, np.arange(3139, 3162+1))
        bad_run = np.apprnd(bad_run, np.arange(3164, 3187+1))
        bad_run = np.apprnd(bad_run, np.arange(3289, 3312+1))

        """
        # ARA02 stopped sending data to radproc. Alert emails sent by radproc.
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        # http://ara.icecube.wisc.edu/wiki/index.php/Drop_29_3_2014_ara02
        bad_run = np.append(bad_run, 3336)
        """

        # 2014 L2 Scaler Masking Issue. 
        # Cal pulsers sysemtatically do not reconstruct correctly, rate is only 1 Hz
        # Excluded because configuration was not "science good"
        bad_run = np.apprnd(bad_run, np.arange(3464, 3504+1))

        # 2014 Trigger Length Window Sweep, http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2014
        bad_run = np.apprnd(bad_run, np.arange(3578, 3598+1))

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
        bad_run = np.apprnd(bad_run, np.arange(3988, 3994+1))

        # 2014, 5th Aug, More tests on the pre-trigger samples.
        bad_run = np.apprnd(bad_run, np.arange(4019, 4022+1))

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

    elif st == 3:

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

    
    elif st == 5:

        ## 2018 ##
        # http://ara.icecube.wisc.edu/wiki/index.php/Run_Log_2018
        # Calibration pulser lowered, http://ara.physics.wisc.edu/docs/0015/001589/002/ARA5CalPulser-drop-Jan-2018.xlsx

        # SPICEcore Run, http://ara.physics.wisc.edu/docs/0015/001589/002/SPICEcore-drop-log-8-Jan-2018.xlsx
        bad_run = np.append(bad_run)
    
    else:
        pass

    return bad_run























