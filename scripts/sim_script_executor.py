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

def script_loader(Key = None, Station = None, Year = None, Data = None, Evt_Range = None):

    # run the chunk code
    module = import_module(f'tools.chunk_{Key}_sim')
    method = getattr(module, f'{Key}_sim_collector')
    if Key == 'mf_noise' or Key == 'mf_noise_debug':
        results = method(Data, Station, Year, Evt_Range)
    else:
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
    if Key == 'mf_noise' or Key == 'mf_noise_debug':
        h5_file_name = f'{Output}{Key}_sim_A{Station}_Evt{Evt_Range[0]}_{Evt_Range[1]}_{data_name}'
    else:
        h5_file_name = f'{Output}{data_name}'
    h5_file_name_out = h5_file_name + '.h5'
    hf = h5py.File(h5_file_name_out, 'w')

    #saving result
    hf.create_dataset('config', data=np.array([Station]), compression="gzip", compression_opts=9)
    for r in results:
        print(r, results[r].shape)
        hf.create_dataset(r, data=results[r], compression="gzip", compression_opts=9)
    hf.close()
    print(f'output is {h5_file_name_out}')

    # quick size check
    size_checker(h5_file_name_out)

    if Key == 'mf':
    
        #link AraRoot
        import ROOT
        ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAra.so")
        ROOT.gSystem.Load(os.environ.get('ARA_UTIL_INSTALL_DIR')+"/lib/libAraEvent.so")

        nu_elst = np.array([0.1, 0.9])
        off_cone = np.arange(0, 4.1, 0.5)
        ant_res = np.arange(0, -61, -10, dtype = int)
        nu_info = ['NuE Nu CC','NuMu Nu NC']

        lag_pad = results['lag_pad'].astype(np.double)
        mf_wf_hit_max = results['mf_wf_hit_max']
        mf_hit_max = results['mf_hit_max']
        mf_hit_max_param = results['mf_hit_max_param']
        mf_dim = mf_wf_hit_max.shape
        """
        file = uproot3.recreate(h5_file_name + '_hit_time.root')
        print(h5_file_name + '_hit_time.root')   
     
        for evt in tqdm(range(mf_dim[0])):
            
            file[f'evt{evt}'] = uproot3.newtree({'time':np.double, 'coef':np.double, 'ant_res':np.double, 'off_cone':np.double, 'in_elst':np.double})
            file[f'evt{evt}'].extend({'time':mf_hit_max[evt,0].astype(np.double), 'coef':mf_hit_max[evt,1].astype(np.double), 
                                        'ant_res':mf_hit_max_param[evt,:,0].astype(np.double), 'off_cone':mf_hit_max_param[evt,:,1].astype(np.double), 'in_elst':mf_hit_max_param[evt,:,2].astype(np.double)})
        """
        mf_hit_cut = results['mf_hit_cut']

        file = uproot3.recreate(h5_file_name + '_multiple_hit_time.root')
        print(h5_file_name + '_multiple_hit_time.root')

        for evt in tqdm(range(mf_dim[0])):
            file[f'evt{evt}'] = uproot3.newtree({'ch0_time':np.double, 'ch0_coef':np.double,
                                                    'ch1_time':np.double, 'ch1_coef':np.double,
                                                    'ch2_time':np.double, 'ch2_coef':np.double,
                                                    'ch3_time':np.double, 'ch3_coef':np.double,
                                                    'ch4_time':np.double, 'ch4_coef':np.double,
                                                    'ch5_time':np.double, 'ch5_coef':np.double,
                                                    'ch6_time':np.double, 'ch6_coef':np.double,
                                                    'ch7_time':np.double, 'ch7_coef':np.double,
                                                    'ch8_time':np.double, 'ch8_coef':np.double,
                                                    'ch9_time':np.double, 'ch9_coef':np.double,
                                                    'ch10_time':np.double, 'ch10_coef':np.double,
                                                    'ch11_time':np.double, 'ch11_coef':np.double,
                                                    'ch12_time':np.double, 'ch12_coef':np.double,
                                                    'ch13_time':np.double, 'ch13_coef':np.double,
                                                    'ch14_time':np.double, 'ch14_coef':np.double,
                                                    'ch15_time':np.double, 'ch15_coef':np.double})
            file[f'evt{evt}'].extend({'ch0_time':mf_hit_cut[evt,0,0].astype(np.double), 'ch0_coef':mf_hit_cut[evt,1,0].astype(np.double),  
                                        'ch1_time':mf_hit_cut[evt,0,1].astype(np.double), 'ch1_coef':mf_hit_cut[evt,1,1].astype(np.double),       
                                        'ch2_time':mf_hit_cut[evt,0,2].astype(np.double), 'ch2_coef':mf_hit_cut[evt,1,2].astype(np.double),       
                                        'ch3_time':mf_hit_cut[evt,0,3].astype(np.double), 'ch3_coef':mf_hit_cut[evt,1,3].astype(np.double),       
                                        'ch4_time':mf_hit_cut[evt,0,4].astype(np.double), 'ch4_coef':mf_hit_cut[evt,1,4].astype(np.double),       
                                        'ch5_time':mf_hit_cut[evt,0,5].astype(np.double), 'ch5_coef':mf_hit_cut[evt,1,5].astype(np.double),       
                                        'ch6_time':mf_hit_cut[evt,0,6].astype(np.double), 'ch6_coef':mf_hit_cut[evt,1,6].astype(np.double),       
                                        'ch7_time':mf_hit_cut[evt,0,7].astype(np.double), 'ch7_coef':mf_hit_cut[evt,1,7].astype(np.double),       
                                        'ch8_time':mf_hit_cut[evt,0,8].astype(np.double), 'ch8_coef':mf_hit_cut[evt,1,8].astype(np.double),       
                                        'ch9_time':mf_hit_cut[evt,0,9].astype(np.double), 'ch9_coef':mf_hit_cut[evt,1,9].astype(np.double),       
                                        'ch10_time':mf_hit_cut[evt,0,10].astype(np.double), 'ch10_coef':mf_hit_cut[evt,1,10].astype(np.double),       
                                        'ch11_time':mf_hit_cut[evt,0,11].astype(np.double), 'ch11_coef':mf_hit_cut[evt,1,11].astype(np.double),       
                                        'ch12_time':mf_hit_cut[evt,0,12].astype(np.double), 'ch12_coef':mf_hit_cut[evt,1,12].astype(np.double),       
                                        'ch13_time':mf_hit_cut[evt,0,13].astype(np.double), 'ch13_coef':mf_hit_cut[evt,1,13].astype(np.double),       
                                        'ch14_time':mf_hit_cut[evt,0,14].astype(np.double), 'ch14_coef':mf_hit_cut[evt,1,14].astype(np.double),       
                                        'ch15_time':mf_hit_cut[evt,0,15].astype(np.double), 'ch15_coef':mf_hit_cut[evt,1,15].astype(np.double)})       
   

        """
        myfile = ROOT.TFile(h5_file_name + '_tgraph.root', 'RECREATE' )
        print(h5_file_name + '_tgraph.root')

        for evt in tqdm(range(mf_dim[0])):
            for ant in range(16):
                            title = f'event_id:{evt}, ch:{ant}, ant_rec:{mf_hit_max_param[evt,ant,0]} off_cone:{mf_hit_max_param[evt,ant,1]} in_elst:{mf_hit_max_param[evt,ant,2]}'
                            gr = ROOT.TGraph(len(lag_pad), lag_pad, mf_wf_hit_max[evt,:,ant].astype(np.double))
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
    evt_range = None
    if key == 'mf_noise' or key == 'mf_noise_debug':
        evt_range = np.asarray(sys.argv[5].split(','), dtype = int)

    script_loader(Key = key, Station = station, Year = year, Data = data, Evt_Range = evt_range)













    
