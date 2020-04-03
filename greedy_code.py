# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:43:17 2020

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
import Utils_generation as utils  

#%%

def mad(xin = 0):
    
    import numpy as np
    
    z = np.median(abs(xin - np.median(xin)))/0.6735
    
    return z 
    
    

    
def threshold_greedy_w( Ai, Arefi, nbr, eps, seuilinit=0):
    
    (nb_obs, nb_pix, nb_sources)=np.shape(Ai)
    
    seuil=np.zeros((nb_pix, nb_sources))
    nbrf=np.zeros((nb_sources))
    norme=LA.norm(ft.dct(Ai-Arefi, axis=1, norm='ortho'), axis=0)
    
    for H in range(nb_sources):
        
        nbrH=int(nbr[H])
        
        normeR=norme[:,H]
        ww=np.zeros((nb_pix))
        
        normeR=norme[:,H]
        normemax=np.max(normeR)
        for j in range(nb_pix):
            ww[j]=eps/(eps+normeR[j]/normemax)
        I = abs(normeR).argsort()[::-1]
        
        if normeR[I[nbrH+1]]>1e-14:
            
            print('new threshold')
            seuili=normeR[I[nbrH+1]]
            nbrfH=nbrH+1
            
        else:
            
            print('same threshold')
            seuili=seuilinit[:,H]
            nbrfH=nbrH
            
        seuil[:,H]=ww*dp(seuili)
        nbrf[H]=nbrfH
        
    return(seuil, nbrf)
    

def threshold_fix(Ai, Arefi, nbr, eps, Weights):
        
    (nb_obs, nb_pix, nb_sources)=np.shape(Ai)
    
    seuil=np.zeros((nb_pix, nb_sources))
   
    norme=LA.norm(ft.dct(Ai-Arefi, axis=1, norm='ortho'), axis=0)
    
    for H in range(nb_sources):
        
        nbrH=int(nbr[H])
        
        normeR=norme[:,H]
        ww=np.zeros((nb_pix))
        
        normeR=norme[:,H]
        normemax=np.max(normeR)
        for j in range(nb_pix):
            ww[j]=eps/(eps+normeR[j]/normemax)
        I = abs(normeR).argsort()[::-1]

            
        seuili=normeR[I[nbrH]]
        
        seuil[:,H]=ww*dp(seuili)
  
    return(seuil)
    