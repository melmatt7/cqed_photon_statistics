import numpy as np
import os
import torch

def genGO(Go, trunc=0):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    N = len(Go)
    if N == 1:
        return np.zeros((1,1))

    for i in range(N-trunc-1, 0, -1):
        if N-trunc-1-i > 0:
            GO_temp = np.hstack((np.zeros((i, N-trunc-1-i)), (Go[N-i:N].reshape(i, 1)), Go[N-i]*np.eye(i)))
            GO = np.vstack((GO, GO_temp))
        else:
            GO_temp = np.hstack(((Go[N-i:N].reshape(i, 1)), Go[N-i]*np.eye(i)))
            GO = GO_temp
        

    return GO
