import numpy as np
import ast
import os
import getpass
import matplotlib.pyplot as plt
import socket

def get_def_path(what="trajectories"):
    user = getpass.getuser()
    uu = socket.gethostname()
    if uu in ["pop-os"]:
        defpath = "/home/{}/qmon_sindy/{}/".format(user,what)
    else:
        defpath = "/data/uab-giq/scratch2/matias/qmon_sindy/{}/".format(what)
    os.makedirs(defpath,exist_ok=True)
    return defpath


def give_params(periods=100, ppp=50, mode="exp-dec"):
    if mode == "exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [200., 1., 0.] ##antes kappa = 0.8
    elif mode =="osc-exp-dec":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1 , 20., [[200.,0.], [-.5, 5.]] ##antes kappa = 0.8
    elif mode =="sin":
        gamma, omega, n, eta, kappa, params_force  = 0.3, 1. , 10., .1 , .2, [[3., 0.], [0.1]] ##antes kappa = 0.8
    elif mode =="FHN":
        a,b = .7, .8
        tau = 12.5
        I = .5
        delay, zoom = 50., 10.
        gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [[.8, 1.], [a,b,I,tau, delay, zoom]] ##antes kappa = 0.8
    else:
        raise NameError("define force!")
    params_force.append(mode)
    data_t = [float(periods), ppp]
    p= [gamma, omega, n, eta, kappa, params_force, data_t]
    return p, str(p)+"/"

def load_data(itraj = 1, what="hidden_state.npy",mode="exp-dec"):
    """
    what can be either "dys.npy", "external_signal.npy", or hidden_state.npy
    """
    params, exp_path = give_params(mode=mode)

    ####
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params

    path = get_def_path()+ exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, float(periods), ppp)
    return np.load(path+what)



def plot_integration(times,x,dy,f,dire,exp_path,ss=20):
    os.makedirs(dire,exist_ok=True)

    plt.figure(figsize=(5,15))
    ax=plt.subplot(711)
    ax.plot(times,x[:,0])
    ax.set_ylabel("q",size=ss)
    ax.tick_params(axis='y', labelcolor="blue")
    ax = ax.twinx()
    ax.plot(times,x[:,1], color="red")
    ax.tick_params(axis='y', labelcolor="red")
    ax.set_ylabel("p",size=ss)

    ax=plt.subplot(712)
    ax.plot(times,dy[:,0])
    ax.set_ylabel(r'$dy_q$',size=ss)

    ax=plt.subplot(713)
    ax.plot(times,f[:,0])
    ax.set_ylabel("f_q",size=ss)
    ax.tick_params(axis='y', labelcolor="blue")
    ax = ax.twinx()
    ax.plot(times,f[:,1], color="red")
    ax.tick_params(axis='y', labelcolor="red")
    ax.set_ylabel("f_p",size=ss)

    plt.savefig(dire+exp_path[:-1]+".png")
    print(dire, exp_path)
    return dire, exp_path
