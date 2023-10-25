import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
from numba import jit
from integration.steps import *

#
#@jit(nopython=True)
def integrate(f,g,x0,times,dt,mode="euler"):
    """
    dx = f(x,t)*dt + g(x,t)*dW
    with dW = N(0,sqrt(dt))
    """
    x = np.zeros((len(x0), len(times)))
    dWs = np.sqrt(dt)*np.random.randn(len(x0), len(times)-1)
    x[:,0] = x0
    if mode=="euler":
        step = Euler
    elif mode=="RK4":
        step = RK4
    for i,t in enumerate(times[:-1]):
        x[:,i+1]=x[:,i] + step(f,g,x,dWs,i,t,dt) 
    return x

