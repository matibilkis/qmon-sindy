import numpy as np
from numba import jit

@jit(nopython=True)
def Euler(f,g,x,dWs,i,t,dt):
    return f(x[:,i],t)*dt + g(x[:,i],t).dot(dWs[:,i])

@jit(nopython=True)
def RK4(f,g,x,dWs,i,t,dt):
    ###https://people.math.sc.edu/Burkardt/cpp_src/stochastic_rk/stochastic_rk.cpp
    #Runge-Kutta Algorithm for the Numerical Integration
    #of Stochastic Differential Equations
    ##
    a21 =   0.66667754298442
    a31 =   0.63493935027993
    a32 =   0.00342761715422#D+00
    a41 = - 2.32428921184321#D+00
    a42 =   2.69723745129487#D+00
    a43 =   0.29093673271592#D+00
    a51 =   0.25001351164789#D+00
    a52 =   0.67428574806272#D+00
    a53 = - 0.00831795169360#D+00
    a54 =   0.08401868181222#D+00

    q1 = 3.99956364361748#D+00
    q2 = 1.64524970733585#D+00
    q3 = 1.59330355118722#D+00
    q4 = 0.26330006501868#D+00

    t1 = t
    x1 = x[:,i]
    dw=dWs[:,i]
    k1 = dt*f(x1,t) + np.sqrt(q1)*g(x1,t).dot(dw)

    t2 = t1 + (a21 * dt)
    x2 = x1 + (a21 * k1)
    k2 = dt * f(x2,t2) + np.sqrt(q2)*g(x2,t2).dot(dw)

    t3 = t1 + (a31 * dt)  + (a32 * dt)
    x3 = x1 + (a31 * k1) + (a32 * k2)
    k3 = dt * f(x3,t3 ) + np.sqrt(q3)*g(x3, t3).dot(dw)

    t4 = t1 + (a41 * dt)  + (a42 * dt)  + (a43 * dt)
    x4 = x1 + (a41 * k1) + (a42 * k2) + (a43 * k3)
    k4 = dt * f( x4,t4) + np.sqrt(q4)*g(x4,t4).dot(dw)

    return (a51 * k1) + (a52 * k2) + (a53 * k3) + (a54 * k4)
