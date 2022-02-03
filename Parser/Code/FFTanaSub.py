#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:51:51 2020

@author: sanoev
"""
#%%
import numpy as np
       
#%% do fft:
def ffthan(data, sr = 4, han = True, pad = True): # last dimension is fft dim
    if han:
        han = np.hanning(data.shape[-1]) 
        data_han = data*han
    else:
        data_han = data
    if pad:
        zv = np.zeros(data_han.shape)    
        data_han_pad = np.concatenate((zv, data_han, zv), axis=-1) 
    else:
        data_han_pad = data
    ft = np.fft.rfft(data_han_pad, axis=-1)
    freqs = np.fft.rfftfreq(data_han_pad.shape[-1], 1/sr)
    #ft = ft[:,(freqs>0) & (freqs < maxfoi)]
    #freqsN = freqs[(freqs>0) & (freqs < maxfoi)]
    ft = abs(ft)**2
    return ft, freqs

def ffthan_wrap(activations, avgHidden = True, evoked = False, z = False, sr = 5):
    act = np.moveaxis(activations, [0,1,2], [0,2,1])
    if evoked == True:
        act = np.mean(act,axis=1)
    if avgHidden == True:
        act = np.mean(act,axis=0)
    ft, freqs = ffthan(act,sr, han = True, pad = True)
    
    #ft, freqs = subfun.ffthan(act, 20)    
    if evoked == False and avgHidden == False:
        ft = np.mean(np.mean(ft, axis=1),axis=0)
    elif  evoked == False or avgHidden == False:
        ft = np.mean(ft, axis=0)
    if z == True:
        ft = (ft-np.mean(ft))/np.std(ft)
        #ft = ft/np.mean(ft)
    return ft, freqs