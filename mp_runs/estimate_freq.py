import os
import multiprocessing as mp

cores = mp.cpu_count()
def int_seed(itraj):
    os.system("python3 numerics/integration/integrate.py --itraj {}".format(itraj))
    print(itraj,"done")
    os.system("python3 numerics/NN/modes/normal.py --itraj {}".format(itraj))
    print(itraj,"training done")

with mp.Pool(cores) as p:
    p.map(int_seed, range(0,5000))
