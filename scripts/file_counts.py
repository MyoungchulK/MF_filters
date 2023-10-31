import sys
from glob import glob
import subprocess

D_PATH = str(sys.argv[1])

# sort
print('Tar path:', D_PATH)

d_list_chaos = glob(f'{D_PATH}/*')
d_len = len(d_list_chaos)
print('Total dirs:',d_len)

for d in d_list_chaos:

    subprocess.run(f'cd {d}; pwd', shell = True) # file path
    subprocess.run(f'cd {d}; ls -1 | wc -l', shell = True) # file counts
    subprocess.run(f'cd {d}; du -sh', shell = True) # total sizes
