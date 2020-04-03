# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:11:12 2020

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

def thr1(Sr, Ntilde,dof, pvalue):
    
    import scipy.special
    
    gammaValue=np.sqrt(2)*scipy.special.gamma((dof+1.)/2.)/scipy.special.gamma(dof/2.)
    
    gstd=np.sqrt(dof-gammaValue**2)
    
    prodN=LA.norm(ft.dct(np.dot(Ntilde, np.diag(Sr)), axis=1, norm='ortho'), axis=0)
    
    sigma_n=mad(prodN)
    
    sigmaF=(float(sigma_n))/float(gstd)
    
    gm=np.median(stats.chi.ppf(pvalue, dof, scale=sigmaF))
    
    
    return(gm)
    
def thr2(Sr, sigma, dof, pvalue):
    
    Sd=ft.dct(np.diag(Sr), axis=1, norm='ortho')
    
    gm=np.median(stats.chi.ppf(pvalue, dof, scale=sigma*LA.norm(Sd, axis=0)))
    
    return(gm)
    
def thr3(Sr, Ntilde,sigma, dof, pvalue):
    
    import scipy.special
    
    g1=thr2(Sr, sigma, dof, pvalue)
    
    gammaValue=np.sqrt(2)*scipy.special.gamma((dof+1.)/2.)/scipy.special.gamma(dof/2.)
    
    gstd=np.sqrt(dof-gammaValue**2)
    
    prodN=LA.norm(ft.dct(np.dot(Ntilde, np.diag(Sr)), axis=1, norm='ortho'), axis=0)
    
    if np.sum(prodN>g1)>1:
        print(np.sum(prodN>g1))
 
        prodF=dp(prodN[prodN>g1])
        sigma_n=mad(prodF)
    
        sigmaF=(float(sigma_n))/float(gstd)
    
        gm=np.maximum(stats.chi.ppf(pvalue, dof, scale=sigmaF), g1)
    
    else:
        gm=dp(g1)
    
    return( gm)
    
def threshold_interm(Option,sigma,Si, Ai, Arefi, X, dof,  Weights,eps, stepg, pvalue):
    
    S=dp(Si)
    A=dp(Ai)
    Aref=dp(Arefi)
    
    Ntilde=X-np.sum(A*S.T, axis=2)
    
    nb_sources, nb_pix=np.shape(Si)
    
    gamma= stepg*1./np.max(Si**2, axis=1)
    
    seuil=np.zeros((np.shape(Si)))
    
    ww=np.zeros((nb_pix, nb_sources))
    
    for H in range(nb_sources):
        
        norme=LA.norm(ft.dct(A[:,:,H]-Aref[:,:,H], axis=1, norm='ortho'), axis=0)
        normemax=np.max(norme)
        for j in range(nb_pix):
            ww[j, H]=eps/(eps+norme[j]/normemax)
        
        if Option==1:
            h=gamma[H]*thr1(S[H,:], Ntilde,dof, pvalue)
            
        elif Option==2:
            h=gamma[H]*thr2(S[H,:], sigma, dof, pvalue)
            
        elif Option==3:
            h=gamma[H]*thr3(S[H,:], Ntilde,sigma, dof, pvalue)       
            
        if Weights:
            seuil[H,:]=h*ww[:, H]
        else:
            seuil[H,:]=h
    
    seuilf=seuil.T
        
    return(seuilf, ww)

def threshold_finalstep(Option,sigma,perc,Si, Ai, Arefi, X, dof,  Weights, stepg=.8, pvalue=.996,eps=1e-3):
    """
    Option : 
        1 : Threshold computed on the MAD operator of the norm of the propagated noise
        
        2 : Threshold based on the statistic of the inupt noise
        
        3 : Threshold based on the MAD operator of the residual of the noise 
        over the noise-dependent threshold
    """
    (nb_obs, nb_pix, nb_sources)=np.shape(Ai)
    
    seuil, ww=threshold_interm(Option,sigma,Si, Ai, Arefi, X, dof,  Weights,eps, stepg, pvalue)
    
    norme=LA.norm(ft.dct(Ai-Arefi, axis=1, norm='ortho'), axis=0)
    for H in range(nb_sources):

        seuil_i=dp(seuil[:, H])
        normeR=norme[:,H]
        if Weights:
            indNZ = np.where(abs(normeR) -seuil_i/ww[:, H] > 0)[0]
        else:
            indNZ = np.where(abs(normeR) -seuil_i > 0)[0]
        if len(indNZ)==0:
            seuil[:,H]=dp(seuil_i)
            print('no elt detected')    
        else:
            I = abs(normeR[indNZ]).argsort()[::-1]
            Kval=np.int(np.floor(perc*len(indNZ)))
            if Kval>len(I)-1 or I[Kval]>len(indNZ)-1 :
                seuil[:,H]=dp(seuil_i)
                print('threshold source-dpdt only')
            else:
                print('threshold based on the nbr of coefs', Kval, len(indNZ))
                IndIX = np.int(indNZ[I[Kval]])
                thr=abs(normeR[IndIX])
                if Weights:
                    seuil[:,H]=ww[:,H]*dp(thr)
                else:
                    seuil[:,H]=dp(thr)
            
    return(seuil)    

#%%
def threshold_oracle(As, Arefi):
   
    (nb_obs, nb_pix, nb_sources)=np.shape(As)
    
    seuil=np.ones((nb_pix, nb_sources))
    
    #norme=LA.norm(ft.dct(Ai-Arefi, axis=1, norm='ortho'), axis=0)
    Anorm_so=LA.norm(ft.dct(As-Arefi, axis=1, norm='ortho'), axis=0)
    for H in range(nb_sources):
        I=np.where(Anorm_so[:,H]>1e-4)
        seuil_i=np.ones((nb_pix))*5e-1
        seuil_i[I]=1e-5
        
        
        seuil[:,H]=dp(seuil_i)
            
    return(seuil)    
