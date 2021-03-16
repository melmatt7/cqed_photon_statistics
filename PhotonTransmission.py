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
from itertools import combinations
from genGO_vec import genGO


class PhotonTransmission:
    
    def __init__(self, plot_params, sim_params):

        #plotting parameters
        self.s    = plot_params['s'] 
        self.lim1 = plot_params['lim1']
        self.lim2 = plot_params['lim2'] 
        self.wnum = plot_params['wnum']
        self.time_val = plot_params['time_val'] 

        #simulation parameters
        self.k   = sim_params['k']        
        self.k1  = sim_params['k1']        
        self.k2  = sim_params['k2']        
        self.go  = sim_params['go']          
        self.g   = sim_params['g']     
        self.Gc  = sim_params['Gc']    
        self.wc  = sim_params['wc']           
        self.we  = sim_params['we']            
        self.gnd = sim_params['gnd']  
        self.N   = sim_params['N']   
        self.jvec = sim_params['jvec']
        self.We = sim_params['We']

        self.w  = np.linspace(self.lim1, self.lim2, num=self.wnum)

        #caching variables
        self.nck0 = None
        self.nck1 = None
        self.nck2 = None
        self.nck1 = None

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

            t[i] = 1j*np.transpose(gnd)*self.a1@self.phi1@self.phi1@D1@self.phi1v@self.phi1v@np.transpose(self.a1)*gnd

        self.T=self.k1*self.k2*t*np.conj(t)

        self.tk=np.sqrt(self.k1*self.k2)*t
        self.T_2port=(self.tk+1)*np.conj(self.tk+1)

        return self.w, self.T, self.T_2port

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


            gw1_diag = torch.diagonal(self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1@self.D1@self.phi1v*np.transpose(self.a1)*self.a1*self.phi1*self.D1,0)

            fw1_diag = torch.diagonal(self.phi1v@self.a2@self.phi2@self.D2@self.phi2v@np.transpose(self.a2)@self.phi1@self.D1@self.phi1v*np.transpose(self.a1)*self.gnd*self.gnd*self.a1@self.phi1,0)

            exp_val = torch.exp(-1j*(self.lambda1-w_val)*self.time_val)

            
            some_val = torch.ones(1,self.N+1) - exp_val

            fw2[i] = torch.dot(fw1_diag, exp_val) + gw1_diag @ some_val.reshape(-1,1)

        val=(self.T**2)*fw2*np.conj(fw2)
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