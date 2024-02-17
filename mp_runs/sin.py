import os
import multiprocessing as mp
from datetime import datetime
import argparse
import numpy as np
import socket
import getpass

user = getpass.getuser()
uu = socket.gethostname()
if uu in ["pop-os"]:
    python = "python3"
else:
    python = "/data/jupyter/software/envs/master/bin/python3.11"


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--itraj", type=int, default=1)
args = parser.parse_args()
global itraj
itraj = args.itraj

alphas = list(np.linspace(0., 1e-2, 7))
def int_seed(alpha):
    os.system("{} numerics/NN/modes/sin/in01234.py --alpha {} --printing 1".format(python,alphas[alpha-1]))

# with mp.Pool(cores) as p:
#     p.map(int_seed, range(0,cores))
int_seed(itraj)
