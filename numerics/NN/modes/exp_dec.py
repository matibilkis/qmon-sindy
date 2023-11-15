import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numerics.utilities.misc import *
import torch
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from numerics.NN.models import *
from numerics.NN.losses import *
from numerics.NN.misc import *
import copy
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    parser.add_argument("--mode", type=str, default="exp-dec")
    
    args = parser.parse_args()
    itraj = args.itraj ###this determines the seed
    mode=args.mode
    printing=False
    
    x = load_data(itraj=itraj, what="hidden_state.npy")
    dy = load_data(itraj=itraj,what="dys.npy")
    f = load_data(itraj=itraj, what="external_signal.npy")


    ####
    params, exp_path = give_params(mode=mode)
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params
    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0,total_time+dt,dt)
    ###

    inputs_cell = [dt,  [gamma, omega, n, eta, kappa, b], [195., -1.5]]


    torch.manual_seed(0)

    dev = torch.device("cpu")
    rrn = RecurrentNetwork(inputs_cell)

    optimizer = torch.optim.Adam(list(rrn.parameters()), lr=1e-2)

    dys = torch.tensor(data=dy, dtype=torch.float32).to(torch.device("cpu"))

    xs_hat, dys_hat, fs_hats = rrn(dys)
    loss = log_lik(dys, dys_hat)
    history = {}
    history["losses"] = [ [loss.item(),err_f(f[:,1],fs_hats)]  ]
    history["params"] = [[k.detach().data for k in list(rrn.parameters())]]
    history["gradients"] = []

    if printing==True:

        print(loss.item())
        print(err_f(f[:,1],fs_hats))
        print(history["params"][-1])
        print("\n")

    for ind in range(3000):
        xs_hat, dys_hat, fs_hats = rrn(dys)
        loss = log_lik(dys, dys_hat, dt=dt)
        loss.backward()
        signal_distance = err_f(f[:,1],fs_hats)
        optimizer.step()


        history["losses"].append([loss.item(),signal_distance] )
        history["params"].append([k.detach().data for k in copy.deepcopy(list(rrn.parameters()))])
        history["gradients"].append(copy.deepcopy([k.grad.numpy() for k in list(rrn.parameters())]))

        if printing==True:
            
            print("**** iteration {} ****".format(ind))
            print(loss.item())
            print(signal_distance)
            print(history["params"][-1])
            print("\n")
        optimizer.zero_grad()
        save_history(history, itraj=itraj, exp_path=exp_path,what="exp_dec_2_params")

        if np.abs(loss.item()) < 1+1e-7:
            break
