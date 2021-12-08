import numpy as np
import os, sys
import h5py
from matplotlib import pyplot as plt
import ROOT


def plot_16_indi(xlabel,ylabel,title
            ,x_data,y_data
            ,d_path,file_name
            ,message
            ,xmin = None, xmax = None
            ,ymin = None, ymax = None
            ,y_scale = None
            ,vpeak = None
            ,absol = None):

    fig = plt.figure(figsize=(24, 18)) # figure size

    ax = fig.add_subplot(111) # want to have a one big XY label for all 16 plots
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel(xlabel, labelpad=30,fontsize=40)
    ax.set_ylabel(ylabel,labelpad=50, fontsize=40)

    plt.title(title, y=1.04,fontsize=35)

    for b in range(16):
    #for b in range(20):

        ax = fig.add_subplot(4,4,b+1)
        #ax = fig.add_subplot(5,4,b+1)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=20)
        #ax.set_xlim(-200,800)
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin,xmax)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin,ymax)
        if y_scale is not None:
            ax.set_yscale('log')
        #ax.set_ylim(-0.8,0.8)
        ax.grid(linestyle=':')
        #ax.set_title(Antenna[b],fontsize=25)
        ax.set_title(f'ch{b}',fontsize=25)

        if absol is not None:
            y_dat = np.abs(y_data[b])
        else:
            y_dat = y_data[b]

        if vpeak is not None:
            ax.plot(x_data[b],y_dat,'-',lw=2,color='red',alpha=0.7,label='Vpeak:'+str(np.round(np.nanmax(np.abs(y_data[b])),2)))
            plt.legend(loc='best',numpoints = 1 ,fontsize=15)
        else:
            ax.plot(x_data[b],y_dat,'-',lw=2,color='red',alpha=0.7)

    plt.tight_layout()

    # saving png into output path
    os.chdir(d_path)
    fig.savefig(file_name,bbox_inches='tight')
    #plt.show()
    plt.close()

    print(message)

def wf_checker(Data, Ped, Station, Run, Output, Sel_evt = None):

    # import cern root and ara root lib from cvmfs
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

    # open a data file
    file = ROOT.TFile.Open(Data)

    # load in the event free for this file
    eventTree = file.Get("eventTree")

    # set the tree address to access our raw data type
    rawEvent = ROOT.RawAtriStationEvent()
    eventTree.SetBranchAddress("event",ROOT.AddressOf(rawEvent))

    # get the number of entries in this file
    num_events = eventTree.GetEntries()
    print('total events:', num_events)

    # open a pedestal file
    calibrator = ROOT.AraEventCalibrator.Instance()
    calibrator.setAtriPedFile(Ped, Station)

    # open general quilty cut
    qual = ROOT.AraQualCuts.Instance()
    
    if Sel_evt is not None:
        pass
    else:
        Sel_evt = np.random.choice(np.arange(10,num_events),1)[0] 
    print('selected event:', Sel_evt)

    # get the desire event
    eventTree.GetEntry(Sel_evt)

    # make a useful event -> calibration process
    usefulEvent = ROOT.UsefulAtriStationEvent(rawEvent,ROOT.AraCalType.kLatestCalib)

    # create output dir
    if not os.path.exists(Output):
        os.makedirs(Output)
    os.chdir(Output)
   
    h5_file_name=f'WF_FFT_A{Station}_R{Run}_E{Sel_evt}.h5'
    hf = h5py.File(h5_file_name, 'w')

    #saving result
    g1 = hf.create_group('Raw_WF')

    # list for plot
    raw_wf_t = []
    raw_wf_v = []
    
    # loop over the antennas
    for ant_i in range(16):

        # TGraph
        if Station == 5:
            if ant_i == 0: ch_i = 0
            if ant_i == 1: ch_i = 1
            if ant_i == 2: ch_i = 8
            if ant_i == 3: ch_i = 9
            if ant_i == 4: ch_i = 16
            if ant_i == 5: ch_i = 17
            if ant_i == 6: ch_i = 24
            if ant_i == 7: ch_i = 25
            if ant_i == 8: ch_i = 2
            if ant_i == 9: ch_i = 3
            if ant_i == 10: ch_i = 10
            if ant_i == 11: ch_i = 11
            if ant_i == 12: ch_i = 18
            if ant_i == 13: ch_i = 19
            if ant_i == 14: ch_i = 26
            if ant_i == 15: ch_i = 27

            graph = usefulEvent.getGraphFromElecChan(ch_i)
        else:
            graph = usefulEvent.getGraphFromRFChan(ant_i)

        # raw wf
        raw_t = np.frombuffer(graph.GetX(),dtype=float,count=-1) # It is ns(nanosecond)
        raw_v = np.frombuffer(graph.GetY(),dtype=float,count=-1) # It is mV
        raw_wf_t.append(raw_t)
        raw_wf_v.append(raw_v)

        # save wf
        g1.create_dataset(f'Ch{ant_i}', data=np.stack([raw_t, raw_v], axis=-1), compression="gzip", compression_opts=9)

    hf.close()  
 
    print(f'output is {Output}{h5_file_name}')
    
    # raw wf plot
    plot_16_indi(r'Time [ $ns$ ]',r'Amplitude [ $mV$ ]',f'Raw WF, A{Station}, Run{Run}, Evt{Sel_evt}'
            ,raw_wf_t, raw_wf_v
            ,Output,f'Raw_WF_A{Station}_R{Run}_E{Sel_evt}.png'
            ,f'Event# {Sel_evt} Raw WF plot was generated!'
            ,vpeak = 'on')

    del Output, h5_file_name
    print('done!')

if __name__ == "__main__":

    # since there is no click package in cobalt...
    if len (sys.argv) !=6 and len (sys.argv) !=7:
        Usage = """
    Usage = python3 %s
    <Raw file ex)/data/exp/ARA/2014/unblinded/L1/ARA02/0116/run002898/event002898.root>
    <Pedestal file ex)/data/exp/ARA/2014/calibration/pedestals/ARA02/pedestalValues.run002894.dat>
    <Station ex)2>
    <Run ex)2898>
    <Output path ex)/home/mkim/>
    if you want choose specific event, 
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
    if len (sys.argv) == 7:   
        sel = int(sys.argv[6])
        wf_checker(Data, Ped, Station, Run, Output, Sel_evt = sel)
        del sel

    else:
        wf_checker(Data, Ped, Station, Run, Output)














    
