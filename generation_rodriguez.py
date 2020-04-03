# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:34:54 2020

@author: ielham
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 13:48:45 2020

@author: ielham
"""

import FrechetMean as fm
import numpy as np 
from scipy import fftpack as ft
from scipy import stats 
import scipy as sp
from copy import deepcopy as dp 
from scipy.fftpack import idct,dct
import scipy.io as sio
from numpy import linalg as LA
import matplotlib.pyplot as plt  
#%%
def VS(t, Energy,lower_freq, higher_freq, activ_param):
    """
    Module special pour la variation du degré de parcimonie, ce code me permet
    d'avoir des VS ayant la même énergie.
    Celle-ci est de sorte à avoir au plus un angle theta; (ce sera le cas
    si j'ai deux coefs actifs)
    """
    ru=0
    
    while ru==0:
        x=np.zeros((t))
        x[lower_freq:higher_freq]=np.array(sp.stats.bernoulli.rvs(activ_param,size=(higher_freq-lower_freq)))
  
        x[0]=0
        ru=np.sum(x!=0)
    x=Energy*x/(LA.norm(x))
        
    res=ft.idct(x, norm='ortho')
    return(res, ru)
    
#%%
def vector_rotated(v1, v2, angle):
    l=len(v1)
    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1,v2) * n1
    n2 = v2 / np.linalg.norm(v2)
    
    a=angle

    I = np.identity(l)

    R = I + ( np.outer(n2,n1) - np.outer(n1,n2) ) * np.sin(a) + ( np.outer(n1,n1) + np.outer(n2,n2) ) * (np.cos(a)-1)
    
    res=np.matmul( R, n1 )
    
    return(res)
#%%
def generate_A_ndim(higher_freq,active_param, thetamax, Energy, t, lower_freq=0,n=2, m=5):

    e1=np.zeros((m))
    
    for j in range(m):
        e1[j]=np.random.uniform(low=0., high=1)
    e1=e1/LA.norm(e1)
    
    ebis=np.random.randn(m)
    
    e2=vector_rotated(e1, ebis,thetamax)
    e2=e2/LA.norm(e2)

    A1=np.zeros((m,2))
    
    A1[:,0]=e1
 
    A2=np.zeros((m, t,n))
    
    VS1, ru1=VS(t, Energy,lower_freq, higher_freq, active_param)
    VS2, ru2=VS(t, Energy, lower_freq, higher_freq, active_param)
    
    VS1u=ft.dct(VS1, norm='ortho')
    VS2u=ft.dct(VS2, norm='ortho')

    ebis1=np.random.randn(m)
    ebis2=np.random.randn(m)
    
    for i in range (t):
        
        theta=VS1[i]
#        ebis=np.random.randn(m)
        
        A2[:,i,0]=vector_rotated(e1, ebis1,theta)
        theta=VS2[i]
       
        A2[:,i,1]=vector_rotated(e2, ebis2,theta)

    #A2=A2/LA.norm(A2, axis=0)
    
    return(A2, e1,e2,VS1u, VS2u, ru1, ru2)
    
def moyFrechet(A2):
    (m,t,n)=np.shape(A2)
    Af1=fm.FrechetMean(A2[:,:,0].reshape((m, t)))
    Af2=fm.FrechetMean(A2[:,:,1].reshape((m, t))) 
    Afre=np.zeros((m,n))

    Afre[:,0]=Af1
    Afre[:,1]=Af2
    
    return(Afre)