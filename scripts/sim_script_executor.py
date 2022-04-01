import numpy as np
import os, sys
import h5py
from importlib import import_module
from tqdm import tqdm
import uproot3

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_utility import size_checker

def script_loader(Key = None, Station = None, Year = None, Data = None):

    # run the chunk code
    module = import_module(f'tools.chunk_{Key}_sim')
    method = getattr(module, f'{Key}_sim_collector')
    results = method(Data, Station, Year)
    del module, method

    # create output dir
    Output = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}_sim/'
    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    slash_idx = Data.rfind('/')
    dot_idx = Data.rfind('.')
    data_name = Data[slash_idx+1:dot_idx]
    del slash_idx, dot_idx
    h5_file_name = f'{Output}{Key}_sim_A{Station}_{data_name}'
    h5_file_name_out = h5_file_name + '.h5'
    hf = h5py.File(h5_file_name_out, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station]), compression="gzip", compression_opts=9)
    for r in results:
        if r == 'mf_wf':
            continue
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    del results, Key, Station, Output
    hf.close()
    print(f'output is {h5_file_name_out}')

    # quick size check
    size_checker(h5_file_name_out)
 
    """
    #link AraRoot
    import ROOT
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAra.so")
    ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

    nu_elst = np.array([0.1, 0.9])
    off_cone = np.arange(0, 4.1, 0.5)
    ant_res = np.arange(0, -61, -10, dtype = int)
    nu_info = ['NuE Nu CC','NuMu Nu NC']

    mf_wf = results['mf_wf']
    print(mf_wf.shape)
    mf_dim = mf_wf.shape
    lag_pad = results['lag_pad'].astype(np.double)

    mf_hit_rec_max = results['mf_hit_rec_max']
    print(mf_hit_rec_max.shape)

    file = uproot3.recreate(h5_file_name + '_hit_time.root')
    print(h5_file_name + '_hit_time.root')   
 
    for evt in tqdm(range(mf_dim[0])):
        
        file[f'evt{evt}'] = uproot3.newtree({'rec0_time':np.double, 'rec0_corr':np.double,
                                            'rec-10_time':np.double, 'rec-10_corr':np.double,
                                            'rec-20_time':np.double, 'rec-20_corr':np.double,
                                            'rec-30_time':np.double, 'rec-30_corr':np.double,
                                            'rec-40_time':np.double, 'rec-40_corr':np.double,
                                            'rec-50_time':np.double, 'rec-50_corr':np.double,
                                            'rec-60_time':np.double, 'rec-60_corr':np.double})
        file[f'evt{evt}'].extend({'rec0_time': mf_hit_rec_max[evt,0,:,0].astype(np.double), 'rec0_corr':mf_hit_rec_max[evt,1,:,0].astype(np.double),
                                'rec-10_time': mf_hit_rec_max[evt,0,:,1].astype(np.double), 'rec-10_corr':mf_hit_rec_max[evt,1,:,1].astype(np.double),
                                'rec-20_time': mf_hit_rec_max[evt,0,:,2].astype(np.double), 'rec-20_corr':mf_hit_rec_max[evt,1,:,2].astype(np.double),
                                'rec-30_time': mf_hit_rec_max[evt,0,:,3].astype(np.double), 'rec-30_corr':mf_hit_rec_max[evt,1,:,3].astype(np.double),
                                'rec-40_time': mf_hit_rec_max[evt,0,:,4].astype(np.double), 'rec-40_corr':mf_hit_rec_max[evt,1,:,4].astype(np.double),
                                'rec-50_time': mf_hit_rec_max[evt,0,:,5].astype(np.double), 'rec-50_corr':mf_hit_rec_max[evt,1,:,5].astype(np.double),
                                'rec-60_time': mf_hit_rec_max[evt,0,:,6].astype(np.double), 'rec-60_corr':mf_hit_rec_max[evt,1,:,6].astype(np.double)})

    
    myfile = ROOT.TFile(h5_file_name + '_tgraph.root', 'RECREATE' )
    print(h5_file_name + '_tgraph.root')

    for evt in tqdm(range(mf_dim[0])):
        for ant in range(16):
            for el in range(len(nu_elst)):
                for off in range(len(off_cone)):
                    for res in range(len(ant_res)):
                        title = f'event_id:{evt}, ch:{ant}, receiving_angle:{ant_res[res]} offcone_angle:{off_cone[off]} {nu_info[el]} inelastiticity:{nu_elst[el]}'
                        gr = ROOT.TGraph(len(lag_pad), lag_pad, mf_wf[evt,:,ant,res,off,el].astype(np.double))
                        gr.SetTitle( title )
                        gr.GetXaxis().SetTitle( 'Lag[ns]' )
                        gr.GetYaxis().SetTitle( 'correlation coefficient' )
                        gr.Write()
    myfile.Close()
    """
    
if __name__ == "__main__":

    if len (sys.argv) < 4:
        Usage = """

    Usage = python3 %s

    <Srtipt Key ex)mf>    
    <Station ex)2>
    <Year ex)2018>
    <Data ex)......> 

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    key=str(sys.argv[1])
    station=int(sys.argv[2])
    year=int(sys.argv[3])
    data=str(sys.argv[4])

    script_loader(Key = key, Station = station, Year = year, Data = data)













    
