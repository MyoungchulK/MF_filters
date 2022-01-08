import numpy as np
import os, sys
from tqdm import tqdm

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def batch_run_loader(Station = None, Output = None):

    batch_info = batch_info_loader(Station)

    print(f'Output path check:{Output}')
    if not os.path.exists(Output):
        os.makedirs(Output)

    batch_info.get_dag_file(Output)    

if __name__ == "__main__":

    if len (sys.argv) < 3:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>
    <Output ex)/home/mkim/analysis/MF_filters/scripts/batch_run/wipac/>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])
    output=str(sys.argv[2])

    batch_run_loader(Station = station, Output = output)
 
