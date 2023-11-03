import numpy as np
import ast
import os
import getpass

def get_def_path():
    user = getpass.getuser()
    if user in ["cooper-cooper","mbilkis"]:
        os.makedirs("/home/{}/qmon_sindy/trajectories/".format(user), exist_ok=True)
        defpath = '/home/{}/qmon_sindy/trajectories/'.format(user)
    elif (user =="matias") or (user == "mati"):
        defpath = '../qmon_sindy/trajectories/'
    elif (user=="giq"):
        defpath = "/media/giq/Nuevo vol/qmon_sindy/trajectories/"
    else:
        raise NameError("check this out")
    return defpath


def give_params():
    #kappa, gamma, omega, n, eta, b = 1., 1e-4, 1e-4, 1e-3, 1e-5, 0.
#    gamma, omega, n, eta, kappa,b  = 15*2*np.pi, 2*np.pi*1e3, 14., 1., 360*2*np.pi, 0. ##Giulio's

    ## I modify a bit the signal-noise ratio
    gamma, omega, n, eta, kappa, params_force  = 15*2*np.pi, 2*np.pi*1e3, 14., 1., 360*2*np.pi, [1e3, 5e1]   ##Giulio's

    p= [gamma, omega, n, eta, kappa, params_force]
    return p, str(p)+"/"

def load_data(itraj = 1, what="hidden_state.npy"):
    """
    what can be either "dys.npy", "external_signal.npy", or hidden_state.npy
    """
    params, exp_path = give_params()

    ####
    gamma, omega, n, eta, kappa, b = params
    period = (2*np.pi/omega)
    total_time = period*10
    dt = period/500

    path = get_def_path()+ exp_path + "{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    return np.load(path+what)
