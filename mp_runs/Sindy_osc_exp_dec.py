import os
import multiprocessing as mp
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

python = "python3"#"/data/jupyter/software/envs/master/bin/python3.11"
def int_seed(itraj, seed=0):
    os.system("{} numerics/integration/external_forces/osc-exp-dec.py --itraj {}".format(python,itraj+seed))
    os.system("{} numerics/NN/modes/Sindy_osc_exp_dec.py --itraj {} --printing 1".format(python,itraj+seed))

# with mp.Pool(cores) as p:
#     p.map(int_seed, range(0,cores))
int_seed(itraj)
