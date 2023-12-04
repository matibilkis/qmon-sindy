import os
import multiprocessing as mp

cores = 2#mp.cpu_count()
def int_seed(itraj):
    os.system("python3 numerics/integration/integrate_fisher.py --itraj {}".format(itraj))
    print(itraj,"done")
    #os.system("python3 numerics/NN/modes/normal.py --itraj {}".format(itraj))
    #print(itraj,"training done")

with mp.Pool(cores) as p:
    p.map(int_seed, range(0,100))#5000))
