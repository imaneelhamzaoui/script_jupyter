# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:52:10 2020

@author: ielham
"""

import numpy as np 
from scipy import fftpack as ft
from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
from scipy.fftpack import idct,dct
import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
import pyStarlet as ps
import initialization as pyl
import BSS_Jerome as bssJ

import Utils_generation as utils  

import algoS as algS
import threshold_strategy as thres
import FISTA as algoA

import stopping_criterion

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import generation_rodriguez as gen
#%%
Se=sio.loadmat('sources')['Se']
A2f=np.zeros((30, 5, 1500, 2))    
Sef=np.zeros((30, 2, 1500))
Aoutf=np.zeros((30, 5, 1500, 2)) 
Soutf=np.zeros((30, 2, 1500))
Xf=np.zeros((30, 5, 1500))
sigma_noisef=np.zeros((30))
#%

Lower_freq=80
Higher_freq=100
#ùù
Se=sio.loadmat('sources')['Se']
ru=1
rho=.5
Energy=1.5
nb_sces=2
nb_obs=5
nb_pix=1500
Angle_btwn=np.pi/3

SNR=65

for R in np.arange(0, 10):

    ru=1
    while ru>0:
        (A2, e1,e2,VS1u, VS2u, ru1, ru2)=gen.generate_A_ndim(Higher_freq,rho, Angle_btwn, Energy, nb_pix, lower_freq=Lower_freq,n=nb_sces, m=nb_obs)
        ru=np.sum(A2<0)
    
    print(ru1)
    print(ru2)
 
    
    
    (X, N, sigma_noise)=utils.XN(A2, Se, m=nb_obs, t=nb_pix, noise_level=SNR)
    #%
    dS={}
    dS['n']=nb_sces
    dS['m']=nb_obs
    dS['t']=nb_pix
    dS['kSMax']=1
    dS['iteMaxXMCA']=5000
    dPatch={}
    dPatch['PatchSize']=500
    dPatch['aMCA']=0
    dPatch['J']=1
    Aout, Sout, Ar, Sr,temp2 ,ss= pyl.GMCAperpatch(X, dS, dPatch, Init=1, aMCA=0)
    
    for r in range(nb_sces):
        Aout[:,:,r] = pyl.Filter1D(x=Aout[:,:,r].squeeze(),J=np.int(np.log2(1536)+1)-1)
    
    
    A2f[R,:,:,:]=dp(A2)
    Aoutf[R,:,:,:]=dp(Aout)
    
    Soutf[R,:,:]=dp(Sout)
    Xf[R,:,:]=dp(X)
    sigma_noisef[R]=dp(sigma_noise)
    
    
    d={}
    d['A2f']=A2f
    d['Aoutf']=Aoutf
    d['Soutf']=Soutf
    #d['Sef']=Sef
    d['Xf']=Xf
    d['sigma_noisef']=sigma_noisef
    d['Se']=Se
    sio.savemat('freq_data_80_100', d)