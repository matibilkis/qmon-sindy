import numpy as np
import ast
import os
import getpass

def get_def_path(what="trajectories"):
    user = getpass.getuser()
    if user in ["cooper-cooper","mbilkis"]:
        os.makedirs("/home/{}/qmon_sindy/{}/".format(user,what), exist_ok=True)
        defpath = '/home/{}/qmon_sindy/{}/'.format(user,what)
    elif (user =="matias") or (user == "mati"):
        defpath = '../qmon_sindy/{}/'.format(what)
    elif (user=="giq"):
        defpath = "/media/giq/Nuevo vol/qmon_sindy/{}/".format(what)
    else:
        raise NameError("check this out")
    return defpath


def give_params(periods=10., ppp=500):
    #kappa, gamma, omega, n, eta, b = 1., 1e-4, 1e-4, 1e-3, 1e-5, 0.
#    gamma, omega, n, eta, kappa,b  = 15*2*np.pi, 2*np.pi*1e3, 14., 1., 360*2*np.pi, 0. ##Giulio's

    ## I modify a bit the signal-noise ratio
    #gamma, omega, n, eta, kappa, params_force  = 15*2*np.pi, 2*np.pi*1e2, 14., 1., 360*2*np.pi, [2e2, 5]   ##Giulio's
    gamma, omega, n, eta, kappa, params_force  = 0.3, 10. , 10., 1.0 , 20., [200., 1., 0.,"exp-dec"] ##antes kappa = 0.8

    data_t = [float(periods), ppp]
    p= [gamma, omega, n, eta, kappa, params_force, data_t]
    return p, str(p)+"/"

def load_data(itraj = 1, what="hidden_state.npy"):
    """
    what can be either "dys.npy", "external_signal.npy", or hidden_state.npy
    """
    params, exp_path = give_params()

    ####
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params

    path = get_def_path()+ exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, float(periods), ppp)
    return np.load(path+what)
