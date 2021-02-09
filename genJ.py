import numpy as np
import torch

def genJ(jvec):
    nck2=len(jvec)
    nck1=round(np.sqrt(2*nck2+0.25)+0.5)

    return jvec[0]*(np.ones((nck1,nck1))-np.eye(nck1))