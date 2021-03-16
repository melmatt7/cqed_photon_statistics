import numpy as np
import torch

def genJ(jvec):
    # print(jvec)
    nck2=len(jvec)
    nck1=round(np.sqrt(2*nck2+0.25)+0.5)
    inds = np.tril_indices(nck1, k=-1)
    B = np.zeros((nck1, nck1))

    if nck2 == 0:
        B[inds] = 0
    else:
        B[inds] = jvec

    # print(B)
    

    return B + np.transpose(B)