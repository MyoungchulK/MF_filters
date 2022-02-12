import os, sys
from subprocess import call

# custom lib
curr_path = os.getcwd()
sys.path.append(curr_path+'/../')
from tools.ara_run_manager import batch_info_loader

def quick_batch_run(Station = None):

    batch_info = batch_info_loader(Station)
    run_list = batch_info.get_dat_list()[0]
    
    for run in run_list:

        CMD=f'python3 script_executor.py trig_ratio {Station} {int(run)}'
        call(CMD.split(' '))

if __name__ == "__main__":

    if len (sys.argv) < 2:
        Usage = """

    If it is data,
    Usage = python3 %s

    <Station ex)2>

        """ %(sys.argv[0])
        print(Usage)
        sys.exit(1)

    # argv
    station=int(sys.argv[1])

    quick_batch_run(Station = station)
 
