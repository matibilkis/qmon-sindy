import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import pickle
import torch

def give_path_model(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    return get_def_path(what=what)+exp_path + "{}itraj/".format(itraj)

def save_history(history,what="NN",exp_path="", itraj=1, periods=10., ppp=500):

    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj)
    os.makedirs(path_model, exist_ok=True)

    with open(path_model+"history.pickle", "wb") as output_file:
       pickle.dump(history, output_file)
    return

def load_history(what="NN",exp_path="",itraj=1, periods=10., ppp=500):
    path_model= give_path_model(what=what,exp_path = exp_path, itraj=itraj)
    with open(path_model+"history.pickle", "rb") as output_file:
       h=pickle.load(output_file)
    return h

def set_params_to_best(rrn, history):
    ll = history["losses"]
    loss_fun = np.array([ll[k][0] for k in range(len(ll))])
    index_favorite = np.argmin(loss_fun)
    news = history["params"][index_favorite]
    with torch.no_grad():

        for j,k in zip(news, list(rrn.parameters())):
            k.data = torch.tensor(j)
    return index_favorite
