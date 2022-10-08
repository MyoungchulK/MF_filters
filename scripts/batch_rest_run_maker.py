import numpy as np
import os, sys
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader
from tools.ara_run_manager import file_sorter

def batch_run_loader(Station = None, Output = None, Analyze_Blind = False, Key = None):

    batch_info = batch_info_loader(Station)

    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    blined = ''
    if Analyze_Blind:
        blined = '_full'
    d_path = os.path.expandvars("$OUTPUT_PATH") + f'/OMF_filter/ARA0{Station}/{Key}{blined}/*'
    d_list, d_run_tot, d_run_range = file_sorter(d_path)

    batch_info.get_rest_dag_file(Output, d_run_range, analyze_blind_dat = Analyze_Blind)    

if __name__ == "__main__":

    if len (sys.argv) < 5:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>
    <Analyze_Blind ex)0 or 1>
    <Key ex)snr>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output=str(sys.argv[2])
    blind=bool(int(sys.argv[3]))
    key=str(sys.argv[4])

    batch_run_loader(Station = station, Output = output, Analyze_Blind = blind, Key = key)
 
