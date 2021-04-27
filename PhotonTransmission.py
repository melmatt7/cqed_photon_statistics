from math import comb
from itertools import combinations

import scipy.linalg as la
import torch
import numpy as np

from utils import genJ
from utils import genGO

class PhotonTransmission:
    
    def __init__(self, plot_params, sim_params):

        #plotting parameters
        self.s    = plot_params['s'] 
        self.lim1 = plot_params['lim1']
        self.lim2 = plot_params['lim2'] 
        self.wnum = plot_params['wnum']
        self.time_val = plot_params['time_val'] 

        #simulation parameters
        self.k       = sim_params['k']        
        self.k1      = sim_params['k1']        
        self.k2      = sim_params['k2']        
        self.go      = sim_params['g']          
        self.g       = sim_params['gamma']     
        self.Gc      = sim_params['Gc']    
        self.wc      = sim_params['wc']           
        self.we      = sim_params['we']            
        self.gnd     = sim_params['gnd']  
        self.N       = sim_params['N']   
        self.jvec    = sim_params['jvec']
        self.We      = sim_params['We']

        self.w  = np.linspace(self.lim1, self.lim2, num=self.wnum)

        #caching variables
        self.nck0 = None
        self.nck1 = None
        self.nck2 = None
        self.nck3 = None

        self.a1 = None
        self.lambda1 = None
        self.phi1 = None
        self.phi1v = None
        self.T = None
        self.T_2port = None
        self.tk = None
        self.D1 = None

        self.a2 = None
        self.lambda2 = None
        self.phi2 = None
        self.phi2v = None
        self.g2_w =None
        self.g2_w_ref = None
        self.D2 = None

        self.a3 = None

        self.first_subspace  = None
        self.second_subspace = None
        self.third_subspace  = None

    def calc_first_subspace(self):
        Go = self.go*torch.ones(self.N)
        G = self.g*torch.ones(self.N)
        # We = np.random.normal(0, self.g*117, 200)
        self.We = self.We[0:self.N]
        # print(We)

        self.nck0 = comb(self.N, 0)
        self.nck1 = comb(self.N, 1)

        # creation/annihilation operators
        gnd = 1
        self.a1 = torch.cat((torch.tensor([1]),torch.zeros(self.N, dtype=torch.cfloat)))

        ## First Excitation Subspace
        Heff1=torch.zeros(self.N+1,self.N+1, dtype=torch.cfloat)
        # cavity loss
        Heff1[0,0]= self.wc - 1j*self.k/2 #+1iw comes from the diagonal contribution later

        if self.N > 0:
            # atomic coupling
            Heff1[0,1:self.N+2] = -1j*Go
            Heff1[1:self.N+2,0] = 1j*Go

            # spontaneous emission
            Heff1[1:self.N+2,1:self.N+2] += (np.diag(self.We)-1j*np.diag(G)/2)

        J=genJ(self.jvec)

        Heff1[1:self.nck0+self.nck1+1,1:self.nck0+self.nck1+1] += J

        lambda1, phi1 = la.eig(Heff1)
        self.lambda1 = torch.from_numpy(lambda1)
        self.phi1 = torch.from_numpy(phi1)
        phi1v = la.inv(phi1)
        self.phi1v = torch.from_numpy(phi1v)


        w_inc = (self.lim2-self.lim1)/self.wnum
        t = torch.zeros(self.wnum, dtype=torch.cfloat)
        for i in range(0,self.wnum):
            w_val = self.lim1+(i+1)*w_inc
            D1 = ((self.lambda1-w_val)**-1)*torch.eye(self.N+1)

            t[i] = 1j*np.transpose(gnd)*self.a1@self.phi1@D1@self.phi1v@np.transpose(self.a1)*gnd

        self.T=self.k1*self.k2*t*np.conj(t)

        self.tk=np.sqrt(self.k1*self.k2)*t
        self.T_2port=(self.tk+1)*np.conj(self.tk+1)

        return self.w, self.T, self.T_2port

    # def calc_first_subspace(self):
        

    #     Go = self.go*torch.ones(self.N)
    #     G = self.g*torch.ones(self.N)
    #     # K = self.k*torch.ones(self.N)
    #     We = self.we*torch.ones(self.N)
        

    #     # values used to dimension the higher excitation hamiltonians
    #     self.nck0 = comb(self.N, 0)
    #     self.nck1 = comb(self.N, 1)

    #     try:
    #         self.nck2 = comb(self.N,2)
    #     except:
    #         self.nck2 = 0

    #     try:
    #         self.nck3 = comb(self.N,3)
    #     except:
    #         self.nck3 = 0

    #     # creation/annihilation operators
    #     gnd = 1
    #     self.a1 = torch.cat((torch.tensor([1]),torch.zeros(self.N, dtype=torch.cfloat)))
    #     # a2 = np.hstack((np.eye(self.N+1), mat.repmat(np.zeros((self.N+1, 1)), 1, nck2)))
    #     # a2[1] = a2[1]*np.sqrt(2)

    #     # a3 = np.zeros((nck0+nck1+nck2,nck0+nck1+nck2+nck3))
    #     # a3[0,0] = np.sqrt(3)
    #     # a3[1:(nck1+nck0),1:(nck1+nck0)] = np.eye(nck1)*(np.sqrt(2))
    #     # a3[nck0+nck1:nck0+nck1+nck2,nck0+nck1:nck0+nck1+nck2] = np.eye(nck2)

    #     # ## Zeroth excitation subspace
    #     # Heff0=0

    #     ## First Excitation Subspace
    #     Heff1=torch.zeros(self.N+1,self.N+1, dtype=torch.cfloat)
    #     # cavity loss
    #     Heff1[0,0]= self.wc - 1j*self.k/2 #+1iw comes from the diagonal contribution later
    #     # jax.ops.index_update(Heff1, jax.ops.index[0,0], wc - 1j*k/2)

    #     if self.N > 0:
    #         # atomic coupling
    #         Heff1[0,1:self.N+2] = Go
    #         # jax.ops.index_update(Heff1, jax.ops.index[0,1:self.N+2], Go)
    #         Heff1[1:self.N+2,0] = Go
    #         # jax.ops.index_update(Heff1, jax.ops.index[1:self.N+2,0], Go)

    #         # spontaneous emission
    #         Heff1[1:self.N+2,1:self.N+2] += (np.diag(We)-1j*np.diag(G)/2)
    #         # jax.ops.index_add(Heff1, jax.ops.index[1:self.N+2,1:self.N+2], (np.diag(We)-1j*np.diag(G)/2))

    #     J=genJ(self.jvec)
    #     Heff1[1:self.nck0+self.nck1+1,1:self.nck0+self.nck1+1] += J
    #     # print(Heff1)

    #     # jax.ops.index_add(Heff1, jax.ops.index[1:nck0+nck1+1,1:nck0+nck1+1], J)
    #     # print(Heff1)
    #     # unsorted phi1
    #     # print(Heff1)
    #     lambda1, phi1 = la.eig(Heff1)
    #     self.lambda1 = torch.from_numpy(lambda1)
    #     self.phi1 = torch.from_numpy(phi1)
    #     # print(phi1)
    #     phi1v = la.inv(phi1)
    #     self.phi1v = torch.from_numpy(phi1v)
    #     # norm=la.norm(phi1, axis=0)
    #     # tol = 10e-5
    #     # a = phi1@np.diag(lambda1)@phi1v
    #     # a.real[abs(a.real) < tol] = 0.0
    #     # a.imag[abs(a.imag) < tol] = 0.0
    #     # print(a)

    #     w_inc = (self.lim2-self.lim1)/self.wnum
    #     t = torch.zeros(self.wnum, dtype=torch.cfloat)
    #     for i in range(0,self.wnum):
    #         w_val = self.lim1+(i+1)*w_inc
    #         D1 = ((self.lambda1-w_val)**-1)*torch.eye(self.N+1)
    #         # print(D1)
    #         # Transmission
    #         # current_time = time.time()

    #         inter3 = torch.matmul(self.a1,self.phi1)
    #         # time3 = time.time()
    #         # print("time3:", time3-current_time)

    #         inter2 = torch.matmul(inter3,D1)
    #         # time2 = time.time()
    #         # print("time2:", time2-time3)

    #         inter1 = torch.matmul(inter2,self.phi1v)
    #         # time1 = time.time()
    #         # print("time1:", time1-time2)

    #         inter0 = torch.matmul(inter1,np.transpose(self.a1))
    #         # time0 = time.time()
    #         # print("time0:", time0-time1)
    #         t[i] = 1j*np.transpose(gnd)*inter0*gnd
    #         # jax.ops.index_update(t, jax.ops.index[i], 1j*np.transpose(gnd)*inter0*gnd)

    #     self.T=self.k1*self.k2*t*np.conj(t)

    #     self.tk=np.sqrt(self.k1*self.k2)*t
    #     self.T_2port=(self.tk+1)*np.conj(self.tk+1)

    #     return self.w, self.T, self.T_2port

    def calc_second_subspace(self):
        Go = self.go*torch.ones(self.N)
        G = self.g*torch.ones(self.N)
        K = self.k*torch.ones(self.N)
        We = self.we*torch.ones(self.N)

        try:
            self.nck2 = comb(self.N,2)
        except:
            self.nck2 = 0
        
        if self.a1 is None:
            _, self.T, self.T_2port = self.calc_first_subspace()
        
        self.a2 = torch.hstack((torch.eye(self.N+1, dtype=torch.cfloat), torch.zeros((self.N+1, 1)).repeat(1,self.nck2)))
        self.a2[0] = self.a2[0]*np.sqrt(2)

        M = int(1+self.N*(self.N+1)/2)
        Heff2=torch.zeros(M,M, dtype=torch.cfloat)
        Heff2[0,0]= 2*(self.wc - 1j*self.k/2)
        Heff2[0,1:self.N+1] = Go*np.sqrt(2)
        Heff2[1:self.N+1,0] = Go*np.sqrt(2)

        if self.N > 0:
            # spontaneous emission
            Heff2[1:self.N+1,1:self.N+1] = (self.wc-1j*self.k/2+(self.we-1j*self.g/2))*torch.eye(self.N)
            comb_vec = torch.tensor([[a[0] + a[1]] for a in combinations((self.we-1j*self.g/2)*torch.ones((self.N,1)),2)])
            Heff2[self.N+1:,self.N+1:] = torch.eye(comb_vec.size(0), dtype=torch.cfloat)*comb_vec

        J=genJ(self.jvec)
        Heff2[self.nck0:self.nck0+self.nck1,self.nck0:self.nck0+self.nck1] += J
        J2 = torch.zeros(comb(self.N,2),comb(self.N,2))
        Heff2[self.nck0+self.nck1:self.nck0+self.nck1+self.nck2,self.nck0+self.nck1:self.nck0+self.nck1+self.nck2] += J2

        GO=genGO(Go)
        r,c=GO.shape
        Heff2[self.N+1:self.N+2+r-1,1:2+c-1] += GO
        Heff2[1:2+c-1,self.N+1:self.N+2+r-1] += np.transpose(GO)

        lambda2, phi2 = la.eig(Heff2)
        self.lambda2 = torch.from_numpy(lambda2)
        self.phi2 = torch.from_numpy(phi2)
        phi2v = la.inv(phi2)
        self.phi2v = torch.from_numpy(phi2v)


        w_inc = (self.lim2-self.lim1)/self.wnum

        fw2 = torch.zeros(self.wnum, dtype=torch.cfloat)
        for i in range(self.wnum):
            w_val = self.lim1+(i+1)*w_inc
            self.D1 = ((self.lambda1-w_val)**-1)*torch.eye(self.N+1)
            self.D2 = ((self.lambda2-2*w_val)**-1)*torch.eye(M)

            # interA = inter3 @ phi1v
            # interB = torch.matmul(interA, a2)
            # interC = torch.matmul(interB, phi2)
            # interD = torch.matmul(interC, D2)
            # interE = torch.matmul(interD, phi2v)
            # interF = torch.matmul(interE, np.transpose(a2))
            # interG = torch.matmul(interF, phi1)
            # interJ = torch.matmul(interG, D1)
            # interH = torch.matmul(interJ, phi1v)
            # interI = torch.matmul(interH,np.transpose(a1))
            gw1_diag = torch.diagonal(self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1@self.D1@self.phi1v*np.transpose(self.a1)*self.a1*self.phi1*self.D1,0)
            # print(self.a2)
            # print(self.Heff1)
            # print(gw1_diag)
            # print(self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1@self.D1@self.phi1v)
            # exit()
            # print(self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1*self.phi1*self.D1*self.phi1v*np.transpose(self.a1)*self.a1*self.phi1*self.D1)
            
            # exit()
            # print(self.lambda1)
            # self.phi1.real[abs(self.phi1.real) < tol] = 0.0
            # self.phi1.imag[abs(self.phi1.imag) < tol] = 0.0
            # print(self.phi1)

            # b = self.D1@self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1
            # b.real[abs(b.real) < tol] = 0.0
            # b.imag[abs(b.imag) < tol] = 0.0
            # print(b)
            fw1_diag = torch.diagonal(self.phi1v@self.a2@self.phi2@self.D2@self.phi2v@np.transpose(self.a2)@self.phi1@self.D1@self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1,0)
            # print(fw1_diag)
            # a = self.phi1v@self.a2@self.phi2@self.D2@self.phi2v@np.transpose(self.a2)@self.phi1
            # a.real[abs(a.real) < tol] = 0.0
            # a.imag[abs(a.imag) < tol] = 0.0
            # print(a)
            # exit()
            # print(self.phi1v@self.a2@self.phi2@self.D2)
            exp_val = torch.exp(-1j*(self.lambda1-w_val)*self.time_val)

            # print(torch.sum(fw1_diag))
            # exit()
            
            some_val = torch.ones(1,self.N+1) - exp_val


            # print(some_val.size())
            # print(np.transpose(gwl_diag).size())
            fw2[i] = torch.dot(fw1_diag, exp_val) + gw1_diag @ some_val.reshape(-1,1)
        #     # jax.ops.index_update(t, jax.ops.index[i], 1j*np.transpose(gnd)*inter0*gnd)

        # print(fw2)
        # exit()
        val=(self.T**2)*fw2*np.conj(fw2)
        # print(fw2[2316])
        self.g2_w=(self.k1**2*self.k2**2)/(self.T**2)*fw2*np.conj(fw2)
        self.g2_w_ref=np.abs(-self.k1*self.k2*fw2+4*(self.tk)+2)**2/(self.T_2port**2)/4

        return self.w, self.T, self.T_2port, self.g2_w, self.g2_w_ref

    def calc_third_subspace(self):

        try:
            self.nck3 = comb(self.N,3)
        except:
            self.nck3 = 0

        if self.a2 is None:
            _, self.T, self.T_2port, self.g2_w, self.g2_w_ref= self.calc_second_subspace()

        self.a3 = np.zeros((self.nck0+self.nck1+self.nck2,self.nck0+self.nck1+self.nck2+self.nck3))
        self.a3[0,0] = np.sqrt(3)
        self.a3[1:(self.nck1+self.nck0),1:(self.nck1+self.nck0)] = np.eye(self.nck1)*(np.sqrt(2))
        self.a3[self.nck0+self.nck1:self.nck0+self.nck1+self.nck2,self.nck0+self.nck1:self.nck0+self.nck1+self.nck2] = np.eye(self.nck2)

        return 0