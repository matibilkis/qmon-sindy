import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())
from scipy.linalg import solve_continuous_are

class GRNN(torch.nn.Module):
    def __init__(self,inputs_cell):
        super(GRNN, self).__init__()

        self.dt, self.simulation_params, trainable_params = inputs_cell
        gamma, omega, n, eta, kappa, b = self.simulation_params
        self.gamma = gamma
        self.omega_p = torch.nn.Parameter(torch.tensor(trainable_params))

        self.A = torch.tensor(data=[[-gamma/2, omega],[-omega,-gamma/2]], dtype=torch.float32).detach()
        self.proj_C = torch.tensor(data=[[1.,0.],[0.,0.]], dtype=torch.float32).detach()
        self.C = np.sqrt(4*eta*kappa)*self.proj_C.detach()
        self.D = (gamma*(n+0.5) + kappa)*torch.eye(2).detach()



    def forward(self, dy, state):
        """
        input_data is dy
        hidden_state is x: (<q>, <p>, Var[x], Var[p], Cov(q,q)})
        output dy_hat
        """
        x = state[:2]
        [vx,vp,cxp] = state[2:5]
        t = state[-1]
        cov = torch.tensor(data = [[vx,cxp],[cxp,vp]], dtype=torch.float32)

        xicov = cov.matmul(self.C.T)
        Adi = .5*torch.tensor([[-self.gamma,0],[0,-self.gamma]],dtype=torch.float32)
        Anon = self.omega_p*torch.tensor([[0,1],[-1,0]],dtype=torch.float32)
        A = Adi+Anon
        dx = (A - xicov.matmul(self.C)).matmul(x)*self.dt + xicov.matmul(dy)

        dcov = self.dt*(cov.matmul(A.T) + (A).matmul(cov) + self.D - (xicov.matmul(xicov.T)))
        ncov = cov+dcov
        nstate = torch.concatenate([(x + dx), torch.tensor([ncov[0,0],ncov[1,1],ncov[1,0]]), torch.tensor([t+self.dt])])
        dy_hat = self.C.matmul(x)*self.dt
        return nstate, dy_hat

class RecurrentNetwork(torch.nn.Module):
    def __init__(self,inputs_cell):
        super(RecurrentNetwork, self).__init__()
        self.RCell = GRNN(inputs_cell=inputs_cell)
        self.dt, self.simulation_params, trainable_params = inputs_cell

    def forward(self, dys):
        dys_hat = []

        ### Find stationary value of covariance for the parameter RCell currently has
        gamma, omega, n, eta, kappa, b = self.simulation_params
        A = np.array([[-gamma/2, omega],[-omega, -gamma/2]])
        proj_C = np.array([[1.,0.],[0.,0.]])
        C = np.sqrt(4*eta*kappa)*proj_C
        D = np.diag([gamma*(n+0.5) + kappa]*2)
        G = np.zeros((2,2))
        Cov = solve_continuous_are((A-G.dot(C)).T, C.T, D- (G.T).dot(G), np.eye(2)) #### A.T because the way it's implemented!
        t0=0.
        xs_hat = [torch.tensor([0., 0., Cov[0,0], Cov[1,1],Cov[1,0], t0], dtype=torch.float32)]
        x_hat = xs_hat[0]
        for dy_t in dys:
            x_hat, dy_hat = self.RCell(dy_t, x_hat)
            dys_hat += [dy_hat]
            xs_hat += [x_hat]
        return torch.stack(xs_hat), torch.stack(dys_hat)
