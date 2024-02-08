import os
import multiprocessing as mp
from datetime import datetime
import argparse
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

cores=7
# python = "/data/jupyter/software/envs/master/bin/python3.11"
python="python3"
alphas = list(np.linspace(0., 1e-2, 7))
def int_seed(alpha):
    # os.system("{} numerics/integration/external_forces/FHN.py --itraj 1".format(python))
    os.system("{} numerics/NN/modes/FHN.py --alpha {} --printing 1".format(python,alphas[alpha-1]))

with mp.Pool(cores) as p:
    p.map(int_seed, range(0,cores))
# int_seed(itraj)
