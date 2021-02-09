import numpy as np
from math import comb
from genGO_vec import genGO

def genGO2(Go, trunc=0):
    N = len(Go)

    if N < 3:
        return []

    for i in range(N-trunc-1, 1, -1):
        print(i)
        if N-trunc-1-i > 0:
            print(np.zeros((comb(i,2), int((N-1-i)*(N+i)/2))))
            GO_temp = np.hstack((np.zeros((comb(i,2), int((N-1-i)*(N+i)/2))), genGO(Go, N-i), Go[N-i]*np.eye(comb(i,2))))
            GO = np.vstack((GO, GO_temp))
        else:
            GO_temp = np.hstack((genGO(Go, N-i), Go[N-i]*np.eye(comb(i,2))))
            GO = GO_temp
        
        print(GO_temp)
        print(GO)

    return GO
