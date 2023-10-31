import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import numpy as np
from tqdm import tqdm
from numba import jit
from scipy.linalg import solve_continuous_are
import argparse

@jit(nopython=True)
def Euler_step_state(x, noise_vector, f):
    return x + A.dot(x)*dt + XiCov.dot(noise_vector) + f*dt

@jit(nopython=True)
def Euler_step_signal(f):
    return f + b*dt

def IntLoop(times):
    N = len(times)
    hidden_state = np.zeros((N,2))
    external_signal = np.zeros((N,2))
    dys = [[0.,0.]]
    for ind, t in enumerate(times[:-1]):
        hidden_state[ind+1] = Euler_step_state(hidden_state[ind], dW[ind], external_signal[ind])
        external_signal[ind+1] = Euler_step_signal(external_signal[ind])
        dys.append(C.dot(hidden_state[ind])*dt + proj_C.dot(dW[ind]))
    return hidden_state, external_signal, dys

def integrate(params, total_time=1, int_step=1e-1, itraj=1, exp_path="",**kwargs):
    global dt, proj_C, A, XiCov, C, dW, b
    dt = int_step
    #### generate long trajectory of noises
    np.random.seed(itraj)
    times = np.arange(0.,total_time+dt,dt)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    gamma, omega, n, eta, kappa, b = params

    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    proj_C = np.array([[1.,0.],[0.,0.]])
    C = np.sqrt(4*eta*kappa)*proj_C
    D = np.diag([gamma*(n+0.5) + kappa]*2)
    G = np.zeros((2,2))

    Cov = solve_continuous_are((A-G.dot(C)).T, C.T, D- (G.T).dot(G), np.eye(2)) #### A.T because the way it's implemented!
    XiCov = Cov.dot(C.T) + G.T

    hidden_state, external_signal, dys = IntLoop(times)

    path = get_def_path() + exp_path + "{}itraj/T_{}_dt_{}/".format(itraj, total_time, dt)
    os.makedirs(path, exist_ok=True)

    if len(times)>1e4:
        indis = np.linspace(0,len(times)-1, int(1e4)).astype(int)
    else:
        indis = np.arange(0,len(times))

    timind = [times[ind] for ind in indis]

    hidden_state =  np.array([hidden_state[ii] for ii in indis])
    external_signal =  np.array([external_signal[ii] for ii in indis])
    dys =  np.array([dys[ii] for ii in indis])

    np.save(path+"hidden_state",hidden_state)
    np.save(path+"external_signal",external_signal)
    np.save(path+"dys",dys)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--itraj", type=int, default=1)
    args = parser.parse_args()

    itraj = args.itraj ###this determines the seed
    params, exp_path = give_params()

    ####
    gamma, omega, n, eta, kappa, b = params
    period = (2*np.pi/omega)
    total_time = period*10
    dt = period/500

    integrate(params=params,
              total_time = total_time,
              int_step = dt,
              itraj=itraj,
              exp_path = exp_path)



#
