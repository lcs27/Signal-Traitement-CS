#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:03:51 2020

@author: gilleschardon
"""

import numpy as np
import numpy.random as rd
import scipy.signal.windows as ww


np.seterr(divide='ignore', invalid='ignore')

def sonar(impulse, variance=1/16):
    
    impulse = np.squeeze(impulse)
    
    gamma = 1/4
    Lmax = 100
    Emax = 1
    
    shift = 50
    Lout = 500
    
    if np.sum(impulse**2) > Emax:
        raise Exception("Énergie trop élevée")
    if impulse.shape[0] > Lmax:
        raise Exception("Impulsion trop longue")
        
    s = np.zeros([Lout])
    
    s[shift:shift + impulse.shape[0]] = gamma * impulse
    
    gwn = rd.normal(size=[2*Lout]) * np.sqrt(variance)
    
    nu_0 = 0.15
    bw = 0.08
    L = 50
    
    t = np.arange(-L, L+1)
    sinc = np.sin(bw*2*np.pi *t) / (bw * 2 * np.pi * t)
    sinc[L] = 1
    sinc = sinc * ww.hann(sinc.shape[0])
    
    sinc = sinc / np.sum(sinc)
    
    f_bp = sinc * np.cos(2*np.pi*nu_0*t) * 2
    f_notch = -f_bp * 0.8
    f_notch[L] = f_notch[L] + 1
    
    noise = np.convolve(gwn, f_notch, mode='full')
    noise = noise[200:200+Lout]
    
    return s + noise
    
def sonar_passif():
    return sonar(np.zeros([0]))
    