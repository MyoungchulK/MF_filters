import numpy as np
import re

def data_info_reader(d_path_str):

    # it must be start with /data/exp/ARA/.....
    # these informations might can salvage from root file itself in future...

    # salvage just number
    d_path = re.sub("\D", "", d_path_str) 

    # year
    yr = int(d_path[:4])
    if yr == 2013:

        # station 
        st = int(d_path[7:9])

        # run 
        run = int(re.sub("\D", "", d_path_str[-11:]))

        # config
        config = config_checker(st, run)

        # month and data
        md = 'Unknown'

        print(f'data info. 1)station:{st} 2)year:{yr} 3)mm/dd:{md} 4)run:{run} 5)config:{config}')

    else:

        # station
        st = int(d_path[5:7])

        # run
        run = int(d_path[-6:])

        # config
        config = config_checker(st, run)

        # month and data
        md = str(d_path[7:11])

        print(f'data info. 1)station:{st} 2)year:{yr} 3)mm/dd:{md} 4)run:{run} 5)config:{config}')

    return st, run, config, yr, md

def config_checker(st, runNum):

    # from Brian: https://github.com/clark2668/a23_analysis_tools/blob/master/tools_Cuts.h

    # default. unknown
    config=0

    # by statation
    if st == 2:
        if runNum>=0 and runNum<=4:
            config=1
        elif runNum>=11 and runNum<=60:
            config=4
        elif runNum>=120 and runNum<=2274:
            config=2
        elif runNum>=2275 and runNum<=3463:
            config=1
        elif runNum>=3465 and runNum<=4027:
            config=3
        elif runNum>=4029 and runNum<=6481:
            config=4
        elif runNum>=6500 and runNum<=8097:
            config=5
        elif runNum>=8100 and runNum<=8246:
            config=4
        else:
            pass

    elif st == 3:
        if runNum>=0 and runNum<=4:
            config=1
        elif runNum>=470 and runNum<=1448:
            config=2
        elif runNum>=1449 and runNum<=1901:
            config=1
        elif runNum>=1902 and runNum<=3103:
            config=5
        elif runNum>=3104 and runNum<=6004:
            config=3
        elif runNum>=6005 and runNum<=7653:
            config=4
        elif runNum>=7658 and runNum<=7808:
            config=3
        else:
            pass

    elif st == 5:
        pass

    return config

def bad_run_checker(run, st):

    bad_run_bool = False

    if run in bad_surface_run(st):
        bad_run_bool = True
        print(f'This run{run}, station{st} is flagged as a bad surface run!')    

    elif run in bad_run(st):
        bad_run_bool = True  
        print(f'This run{run}, station{st} is flagged as a bad run!')

    else:
        pass

    return bad_run_bool

def bad_surface_run(st):

    # masked run(2014~2016) from brian's analysis
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L782

    # array for bad run
    bad_run = np.array([0], dtype=int)

    if st == 2:

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
       
    elif st == 3:

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

    return bad_run[1:] 

def bad_run(st):

    # masked run(2014~2016) from brian's analysis 
    # https://github.com/clark2668/a23_analysis_tools/blob/a7093ab2cbd6b743e603c23b9f296bf2bcce032f/tools_Cuts.h#L881
    
    # array for bad run
    bad_run = np.array([0], dtype=int)

    if st == 2:

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

    return bad_run[1:]























