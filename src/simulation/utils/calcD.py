import torch
import numpy as np

def calc_D (x, y, z, We, g):
    N = len(x)
    # M = int(N*(N-1)/2)
    # lw = torch.zeros(1, M)    
    j, i = np.tril_indices(N, -1, N)
    # print(j)
    # print(i)
    d = np.sqrt(np.power((x[i]-x[j]),2) + np.power((y[i]-y[j]),2) + np.power((z[i]-z[j]),2))
    lw_overlap = 4*g**2/(np.power(We[i] - We[j],2)+4*g**2)
    costheta = np.divide(np.transpose(y[i]-y[j]), np.transpose(d))

    return np.transpose(d), costheta, np.transpose(lw_overlap)