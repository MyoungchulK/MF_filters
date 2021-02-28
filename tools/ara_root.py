import numpy as np
import os
from tools.array import arr_2d

# import root and ara root lib
def ara_root_lib():
    
    # general cern root lib
    import ROOT

    # cvmfs ara lib
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

    return ROOT

# general data loading procedure by araroot
def ara_raw_to_qual(ROOT, data, ped, st):

    # open a data file
    file = ROOT.TFile.Open(data)

    # load in the event free for this file
    evtTree = file.Get("eventTree")
    #del file

    # set the tree address to access our raw data type
    rawEvt = ROOT.RawAtriStationEvent()
    evtTree.SetBranchAddress("event",ROOT.AddressOf(rawEvt))

    # get the number of entries in this file
    num_evts = evtTree.GetEntries()
    print('total events:', num_evts)

    # open a pedestal file
    cal = ROOT.AraEventCalibrator.Instance()
    cal.setAtriPedFile(ped, st)

    # open general quilty cut
    q = ROOT.AraQualCuts.Instance()

    return file, evtTree, rawEvt, num_evts, cal, q #not sure need to return the 'cal'

def useful_evt_maker(ROOT, evtTree, rawEvt, evt, cal): #not sure need to input the 'cal'

    # get the event
    evtTree.GetEntry(evt)

    return ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)

#def useful_evt_maker_del(ROOT):
#
#    ROOT.~UsefulAtriStationEvent()

# make a useful event
#usefulEvt = ROOT.UsefulAtriStationEvent(rawEvt,ROOT.AraCalType.kLatestCalib)
#return usefulEvt

def trig_checker(rawEvt):

    # default trig tag
    trig = -1#'Unknown'

    # trigger tag check
    if rawEvt.isSoftwareTrigger() == 1:
        trig = 2#'Soft'
    elif rawEvt.isCalpulserEvent() == 1:
        trig = 1#'Cal'
    elif rawEvt.isSoftwareTrigger() == 0 and rawEvt.isCalpulserEvent() == 0:
        trig = 0#'RF'
    else:
        pass

    return trig

def qual_checker(q, usefulEvt):

    # default qual tag
    qual = -1#'Unknown'

    # qual tag check
    if q.isGoodEvent(usefulEvt) == 1:
        qual = 1#'Pass'
    elif q.isGoodEvent(usefulEvt) == 0:
        qual = 0#'Cut'
    else:
        pass

    return qual

def ant_xyz(ROOT, Station, num_ant, Years = None):

    # create a geomtool
    geomTool = ROOT.AraGeomTool.Instance()

    # array for xyz coord
    ant_xyz = arr_2d(num_ant, 3, np.nan, float)

    # the x-y coordinates of channels 0-3 are enough for a top down view
    for ant in range(num_ant):

        if Years is not None:
            ant_xyz[ant] = geomTool.getStationInfo(Station,Years).getAntennaInfo(ant).antLocation
        else:
            ant_xyz[ant] = geomTool.getStationInfo(Station).getAntennaInfo(ant).antLocation

    del geomTool

    return ant_xyz









