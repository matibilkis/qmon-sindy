import numpy as np
import math
import matplotlib.pyplot as plt


p=0.9
N = 100
np.sum([  math.comb(N,k)*(p**k)*(1-p)**(N-k)*k  for k in range(0,N)])
