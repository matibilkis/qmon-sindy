import numpy as np
import os
%load_ext autoreload
%autoreload 2
os.getcwd()
os.chdir("/home/mati/qmon-sindy")
from numerics.utilities.misc import *
import matplotlib.pyplot as plt
from numerics.NN.models.sindy_osc_exp_dec import *
from numerics.NN.misc import *
import torch
import numpy as np
from scipy.linalg import solve_continuous_are
from tqdm import tqdm


def load_and_plot(itraj=1, mode="sin"):

    x = load_data(itraj=itraj, what="hidden_state.npy",mode=mode)
    dy = load_data(itraj=itraj,what="dys.npy",mode=mode)
    f = load_data(itraj=itraj, what="external_signal.npy",mode=mode)

    params, exp_path = give_params(mode=mode)
    gamma, omega, n, eta, kappa, b, [periods, ppp] = params
    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0,total_time+dt,dt)

    plot_integration(times,x,dy,f,dire,exp_path)



load_and_plot()


load_and_plot()




load_and_plot()



load_and_plot()





load_and_plot()



































#
