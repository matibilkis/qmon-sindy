import os
import multiprocessing as mp
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

cores = mp.cpu_count()
print(cores)
os.system("/data/jupyter/software/envs/master/bin/python3.11 numerics/integration/integrate.py --itraj {}".format(itraj))
# def int_seed(seed):
#     for k in range(Nstep):
#         os.system("python3 numerics/integration/integrate.py --itraj {}".format(itraj+k))
#         os.system("python3 numerics/NN/run_torch.py --itraj {}".format(itraj+k))
#
#         print(itraj, (datetime.now() - st).seconds)
#
# Nstep = cores-1
# with mp.Pool(cores-1) as p:
#     p.map(int_seed, range(0,40, Nstep))
