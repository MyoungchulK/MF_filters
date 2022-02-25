import numpy as np
from tqdm import tqdm

def ray_corr_collector(Data, Ped, analyze_blind_dat = False):

    Ped = '0'

    print('Collecting reco starts!')

    from tools.ara_data_load import ara_uproot_loader
    from tools.ara_data_load import ara_root_loader
    from tools.ara_constant import ara_const
    from tools.ara_quality_cut import qual_cut_loader

    # geom. info.
    ara_const = ara_const()
    num_ants = ara_const.USEFUL_CHAN_PER_STATION
    del ara_const

    # data config
    ara_uproot = ara_uproot_loader(Data)
    ara_uproot.get_sub_info()
    ara_root = ara_root_loader(Data, Ped, ara_uproot.station_id, ara_uproot.year)
    num_evts = ara_uproot.num_evts
    evt_num = ara_uproot.evt_num
    trig_type = ara_uproot.get_trig_type()

    # qulity cut
    ara_qual = qual_cut_loader(analyze_blind_dat = analyze_blind_dat, verbose = True)
    total_qual_cut = ara_qual.load_qual_cut_result(ara_uproot.station_id, ara_uproot.run)
    qual_cut_sum = np.nansum(total_qual_cut, axis = 1)
    del ara_qual

    clean_evt_idx = np.logical_and(qual_cut_sum == 0, trig_type == 1)
    clean_evt = evt_num[clean_evt_idx]   
    print(f'Number of clean event is {len(clean_evt)}') 
    del qual_cut_sum

    import ROOT
    import os
    #link AraRoot
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraCorrelator.so")
    theCorrelator = ROOT.RayTraceCorrelator(int(ara_uproot.station_id), int(num_ants), 41, 1, '.', '.')
    from ROOT import gInterpreter
    gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/misc_build/include/FFTtools.h"')
    ROOT.gSystem.Load("/cvmfs/ara.opensciencegrid.org/trunk/centos7/misc_build/lib/libRootFftwWrapper.so.3.0.1")
    
    
    """from tools.ara_data_load import ara_geom_loader
    ara_geom = ara_geom_loader(ara_uproot.station_id, ara_uproot.year, verbose = True)
    geomTool = ara_geom.geomTool
    pol_type = ara_geom.get_pol_ch_idx()
    from tools.ara_known_issue import known_issue_loader
    known_issue = known_issue_loader(ara_uproot.station_id)
    bad_ant = known_issue.get_bad_antenna(ara_uproot.run, print_ant_idx = True)

    pairs = theCorrelator.SetupPairs(int(ara_uproot.station_id), geomTool, int(pol_type[0]), bad_ant)
    h_pairs = theCorrelator.SetupPairs(int(ara_uproot.station_id), geomTool, int(1), bad_ant)
    print(pairs.extend(h_pairs))
    """
    print(np.where(evt_num == 452))
    print(trig_type[54])

    # get entry and wf
    ara_root.get_entry(54)
    ara_root.get_useful_evt(ara_root.cal_type.kLatestCalib)
    
    raw_t, raw_v = ara_root.get_rf_ch_wf(0)
    gr_0 = ara_root.gr
    int_gr_0 = ROOT.FFTtools.getInterpolatedGraph(gr_0, 0.5)
    ara_root.del_TGraph() 
    
    raw_t, raw_v = ara_root.get_rf_ch_wf(3)
    gr_3 = ara_root.gr
    int_gr_3 = ROOT.FFTtools.getInterpolatedGraph(gr_3, 0.5)
    ara_root.del_TGraph()

    #corr_nonorm = theCorrelator.getCorrelation_NoNorm(4096, int_gr_0, int_gr_3)
    corr_nor = theCorrelator.getCorrelationGraph_WFweight(int_gr_0, int_gr_3)
    #corr_nor, corr, nor1, nor2, nor12 = theCorrelator.getCorrelationGraph_WFweight(int_gr_0, int_gr_3)
    #corr_nor = cor.grCor
    #corr = cor[1]
    #nor1 = cor[2]
    #nor2= cor[3]
    #nor12 = cor[4]   
 
    corr_nor_all = np.full((corr_nor.GetN(),2), np.nan, dtype = float) 
    corr_nor_all[:,0] = np.frombuffer(corr_nor.GetX(),dtype=float,count=-1)
    corr_nor_all[:,1] = np.frombuffer(corr_nor.GetY(),dtype=float,count=-1)
    """
    corr_all = np.full((corr.GetN(),2), np.nan, dtype = float)
    corr_all[:,0] = np.frombuffer(corr.GetX(),dtype=float,count=-1)
    corr_all[:,1] = np.frombuffer(corr.GetY(),dtype=float,count=-1)

    nor1_all = np.full((nor1.GetN(),2), np.nan, dtype = float)
    nor1_all[:,0] = np.frombuffer(nor1.GetX(),dtype=float,count=-1)
    nor1_all[:,1] = np.frombuffer(nor1.GetY(),dtype=float,count=-1)

    nor2_all = np.full((nor2.GetN(),2), np.nan, dtype = float)
    nor2_all[:,0] = np.frombuffer(nor2.GetX(),dtype=float,count=-1)
    nor2_all[:,1] = np.frombuffer(nor2.GetY(),dtype=float,count=-1)

    nor12_all = np.full((nor12.GetN(),2), np.nan, dtype = float)
    nor12_all[:,0] = np.frombuffer(nor12.GetX(),dtype=float,count=-1)
    nor12_all[:,1] = np.frombuffer(nor12.GetY(),dtype=float,count=-1)
    """
    print('Reco collecting is done!')

    return {'evt_num':evt_num,
            'clean_evt':clean_evt,
            'trig_type':trig_type,
            'corr_t':corr_nor_all[:,0],
            'corr_v':corr_nor_all[:,1]}
            #'corr_nor_all':corr_nor_all,
            #'corr_all':corr_all,
            #'nor1_all':nor1_all,
            #'nor2_all':nor2_all,
            #'nor12_all':nor12_all}







