import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import pickle


def give_path_model(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    return get_def_path(what=what)+exp_path + "{}itraj/periods_{}_ppp_{}/".format(1, float(periods), ppp)

def save_history(history,what="NN",exp_path="", itraj=1, periods=10., ppp=500):

    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj, periods=periods,ppp=ppp)
    os.makedirs(path_model, exist_ok=True)

    with open(path_model+"history.pickle", "wb") as output_file:
       pickle.dump(history, output_file)
    return

def load_history(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj, periods=periods,ppp=ppp)
    with open(path_model+"history.pickle", "rb") as output_file:
       h=pickle.load(output_file)
    return h
