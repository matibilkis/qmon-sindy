import os
import multiprocessing as mp


cores = mp.cpu_count()
def int_seed(seed):
    for k in range(Nstep):
        os.system("python3 numerics/integration/integrate.py --itraj {}".format(itraj+k))
        print(f"{k}, {seed}, done")
#
#
Nstep = cores-1
# int_seed(1)
with mp.Pool(cores-1) as p:
    p.map(int_seed, range(0,40, Nstep))
