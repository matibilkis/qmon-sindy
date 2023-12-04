import os
import sys
sys.path.insert(0, os.getcwd())
from numerics.utilities.misc import *
import numpy as np
from tqdm import tqdm
from numba import jit
from scipy.linalg import solve_continuous_are
import argparse
from numerics.integration.steps import Ikpw
from scipy.linalg import block_diag

@jit(nopython=True)
def Robler_step(t, Yn, Ik, Iij, dt, f,G, d, m):
    """
    https://pypi.org/project/sdeint/
    https://dl.acm.org/doi/abs/10.1007/s10543-005-0039-7
    """
    fnh = f(Yn, t,dt)*dt # shape (d,)
    xicov = Gn = G()
    sum1 = np.dot(Gn, Iij)/np.sqrt(dt) # shape (d, m)

    H20 = Yn + fnh # shape (d,)
    H20b = np.reshape(H20, (d, 1))
    H2 = H20b + sum1 # shape (d, m)

    H30 = Yn
    H3 = H20b - sum1
    fn1h = f(H20, t, dt)*dt
    Yn1 = Yn + 0.5*(fnh + fn1h) + np.dot(xicov, Ik)
    return Yn1

@jit(nopython=True)
def Euler_step_state(x, noise_vector, f):
    return x + A.dot(x)*dt + XiCov.dot(noise_vector) + f*dt

@jit(nopython=True)
def Euler_step_signal(f):
    #prams_foce = [omega, gamma]  linear oscillator
    return f + signal_coeff.dot(f)*dt

######### ROSLER ########
@jit(nopython=True)
def Fhidden(s, t, dt):
    """
    """
    x = s[:2]
    x_th = s[2:4]
    x_dot = np.dot(A,x)
    x_th_dot = np.dot(A - XiCovC, x_th) + np.dot(A_th,x)
    return np.array(list(x_dot) + list(x_th_dot))

######### ROSLER ########
@jit(nopython=True)
def Ghidden():
    return big_XiCov


def IntLoop(times):
    N = len(times)
    hidden_state = np.zeros((N,4))
    external_signal = np.zeros((N,2))
    dys = [[0.,0.]]
    _,I=Ikpw(dW,dt) ### ROSLER
    for ind, t in enumerate(times[:-1]):
        #hidden_state[ind+1] = Euler_step_state(hidden_state[ind], dW[ind], external_signal[ind])
        hidden_state[ind+1] = Robler_step(t, hidden_state[ind], dW[ind,:], I[ind,:,:], dt, Fhidden, Ghidden, 4, 4)
    #    external_signal[ind+1] = Euler_step_signal(external_signal[ind])
        dys.append(C.dot(hidden_state[ind][:2])*dt + proj_C.dot(dW[ind,:2]))
    return hidden_state, external_signal, dys

def integrate(params, periods=10,ppp=500,  itraj=1, exp_path="",**kwargs):
    global dt, proj_C, A, XiCov, C, dW, params_force, signal_coeff,f0, big_XiCov, XiCovC, A_th
    gamma, omega, n, eta, kappa, params_force = params

    f0 = params_force[0]

    period = (2*np.pi/omega)
    total_time = period*periods
    dt = period/ppp
    times = np.arange(0.,total_time+dt,dt)

    #### generate long trajectory of noises
    np.random.seed(itraj)
    dW = np.sqrt(dt)*np.random.randn(len(times),2)
    dW = np.concatenate([dW]*2, axis=1)

    A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
    proj_C = np.array([[1.,0.],[0.,0.]])
    C = np.sqrt(4*eta*kappa)*proj_C
    D = np.diag([gamma*(n+0.5) + kappa]*2)
    G = np.zeros((2,2))
    signal_coeff = np.array([[-params_force[1], params_force[2]],[-params_force[2], -params_force[1]]])


    Cov = solve_continuous_are((A-G.dot(C)).T, C.T, D- (G.T).dot(G), np.eye(2)) #### A.T because the way it's implemented!
    XiCov = Cov.dot(C.T) + G.T
    cov_st = Cov


    D_th = np.eye(2)*0. ## we estimate \omega
    A_th = np.array([[0.,1.],[-1.,0.]])
    cov_st_th = solve_continuous_are( (A-np.dot(cov_st,np.dot(C.T,C))).T, np.eye(2)*0., D_th + np.dot(A_th, cov_st) + np.dot(cov_st, A_th.T), np.eye(2))


    XiCov_th  = np.dot(cov_st_th, C.T) #I take G = 0.
    big_XiCov = block_diag(XiCov, XiCov_th)
    XiCovC = np.dot(XiCov, C)

    hidden_state, external_signal, dys = IntLoop(times)

    path = get_def_path() + exp_path + "{}itraj/periods_{}_ppp_{}/".format(itraj, periods, ppp)
    os.makedirs(path, exist_ok=True)

    if len(times)>1e8:
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
    params, exp_path = give_params(mode="normal")

    ####
    gamma, omega, n, eta, kappa, params_force, [periods, ppp] = params

    integrate(params=params[:-1],
              periods= periods,
              ppp=ppp,
              itraj=itraj,
              exp_path = exp_path)



#
