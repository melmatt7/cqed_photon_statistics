import numpy as np
import jax
from math import comb
import numpy.matlib as mat
import scipy.linalg as la
from scipy.sparse import diags
import os
import time 
from genJ import genJ
import torch
# import matplotlib as 

class PhotonTransmission:
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    
    def __init__(self):
        # config[Hefft] = 0
        # config[g3fig] = 0
        # config[g3time] = 0
        # config[mu] = 0
        # config[sigma] = 0
        # config[tnum] = 2
        # config[t_end] = 20

        #plotting parameters
        plotting[s] = 0.2
        plotting[lim1] = -80
        plotting[lim2] = 80
        plotting[wnum] = 5000

        params[k]  = 43      # cavity linewidth (4.3 - Q~100,000, 43 - Q~10,000) 
        params[k1] = k/2
        params[k2] = k/2
        params[go] = 0.32    # rabi frequency
        params[g]  = 0.043   # emitter linewidth
        params[Gc] = 0       # additional loss channel
        params[wc] = 0       # cavity detuning
        params[we] = 0       # emitter detuning
        params[J]  = 0       # dipole-dipole coupling
        
        self.w  = np.linspace(lim1, lim2, num=wnum)

        # minval = 6.66
        # time = 0

        N = num




    jvec = J*torch.ones(int(N*(N-1)/2))

    Go = go*torch.ones(N)
    G = g*torch.ones(N)
    K = k*torch.ones(N)
    We = we*torch.ones(N)
    

    # values used to dimension the higher excitation hamiltonians
    nck0 = comb(N, 0)
    nck1 = comb(N, 1)

    try:
        nck2 = comb(N,2)
    except:
        nck2 = 0

    try:
        nck3 = comb(N,3)
    except:
        nck3 = 0

    # creation/annihilation operators
    gnd = 1
    a1 = torch.cat((torch.tensor([1]),torch.zeros(N, dtype=torch.cfloat)))
    # a2 = np.hstack((np.eye(N+1), mat.repmat(np.zeros((N+1, 1)), 1, nck2)))
    # a2[1] = a2[1]*np.sqrt(2)

    # a3 = np.zeros((nck0+nck1+nck2,nck0+nck1+nck2+nck3))
    # a3[0,0] = np.sqrt(3)
    # a3[1:(nck1+nck0),1:(nck1+nck0)] = np.eye(nck1)*(np.sqrt(2))
    # a3[nck0+nck1:nck0+nck1+nck2,nck0+nck1:nck0+nck1+nck2] = np.eye(nck2)

    ## Zeroth excitation subspace
    Heff0=0

    ## First Excitation Subspace
    Heff1=torch.zeros(N+1,N+1, dtype=torch.cfloat)
    # cavity loss
    Heff1[0,0]= wc - 1j*k/2 #+1iw comes from the diagonal contribution later
    # jax.ops.index_update(Heff1, jax.ops.index[0,0], wc - 1j*k/2)

    if N > 0:
        # atomic coupling
        Heff1[0,1:N+2] = Go
        # jax.ops.index_update(Heff1, jax.ops.index[0,1:N+2], Go)
        Heff1[1:N+2,0] = Go
        # jax.ops.index_update(Heff1, jax.ops.index[1:N+2,0], Go)

        # spontaneous emission
        Heff1[1:N+2,1:N+2] += (np.diag(We)-1j*np.diag(G)/2)
        # jax.ops.index_add(Heff1, jax.ops.index[1:N+2,1:N+2], (np.diag(We)-1j*np.diag(G)/2))

    J=genJ(jvec)
    Heff1[1:nck0+nck1+1,1:nck0+nck1+1] += J
    # jax.ops.index_add(Heff1, jax.ops.index[1:nck0+nck1+1,1:nck0+nck1+1], J)

    # unsorted phi1
    # print(Heff1)
    lambda1, phi1 = la.eig(Heff1)
    lambda1 = torch.from_numpy(lambda1)
    phi1 = torch.from_numpy(phi1)
    # print(phi1)
    phi1v = la.inv(phi1)
    phi1v = torch.from_numpy(phi1v)
    # norm=la.norm(phi1, axis=0)
    # print(norm)

    w_inc = (lim2-lim1)/wnum
    t = torch.zeros(wnum, dtype=torch.cfloat)
    for i in range(0,wnum):
        w_val = lim1+(i+1)*w_inc
        D1 = ((lambda1-w_val)**-1)*torch.eye(N+1)
        # print(D1)
        # Transmission
        # current_time = time.time()

        inter3 = torch.matmul(a1,phi1)
        # time3 = time.time()
        # print("time3:", time3-current_time)

        inter2 = torch.matmul(inter3,D1)
        # time2 = time.time()
        # print("time2:", time2-time3)

        inter1 = torch.matmul(inter2,phi1v)
        # time1 = time.time()
        # print("time1:", time1-time2)

        inter0 = torch.matmul(inter1,np.transpose(a1))
        # time0 = time.time()
        # print("time0:", time0-time1)
        t[i] = 1j*np.transpose(gnd)*inter0*gnd
        # jax.ops.index_update(t, jax.ops.index[i], 1j*np.transpose(gnd)*inter0*gnd)

    T=k1*k2*t*np.conj(t)

    tk=np.sqrt(k1*k2)*t
    T_2port=(tk+1)*np.conj(tk+1)

    return w, T, T_2port