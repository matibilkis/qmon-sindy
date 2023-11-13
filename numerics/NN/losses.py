import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())

def log_lik(dys, dys_hat, dt=1e-3):
    return torch.sum((dys-dys_hat)**2)/(dt*len(dys))

def err_f(f,fhat, one_dim=True):
    if one_dim==True:
        return np.sum(np.abs(f - fhat[:-1].detach().numpy() ))/np.sum(np.abs(f))
    else:
        return np.sum(np.abs(f - fhat[:-1,:].detach().numpy() ))/np.sum(np.abs(f))
