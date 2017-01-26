#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:49:03 2017

@author: weir
"""

import numpy as _np
import scipy.signal as _dsp
import os as _os
import pybaseutils.utils as _ut
import pybaseutils.fft_analysis as _fft

data_dir = _os.path.abspath(_os.path.join('..','bin'))


def open_datashot(calshot):        
    file_cece = _os.path.join(data_dir, 'CECE.%5i'%(int(calshot),))
    file_ece = _os.path.join(data_dir, 'ECEFAST.%5i'%(int(calshot),))
                              
    CECE = _np.loadtxt(file_cece, dtype=_np.float64, unpack=True)
    ECE = _np.loadtxt(file_ece, dtype=_np.float64, unpack=True)

    # downsample the fast digitized correlation ECE data (1 MHz) down to the 
    # profile measuring systems digitization rate (50 KHz)
    CECE = _fft.downsample(CECE, 1.0/(CECE[2,0]-CECE[1,0]), 
                           1.0/(ECE[2,0]-ECE[1,0]), plotit=False)
    
#    # concatenate the data for outputting
#    nch_e = _np.size(CECE, axis=1) - 1  # number of ECE channels
#    nch_c = _np.size(ECE, axis=1) - 1   # number of CECE channels
#    nt = _np.size(ECE, axis=0)
#    data = _np.zeros((nt, nch_e + nch_c), dtype=_np.float64)
#
#    data[:, 0:nch_c] = _ut.interp(CECE[:,0], CECE[:, 1:], ei=None, xo=ECE[:, 0])
#    data[:, nch_c:] = ECE[:, 1:]

    tt = ECE[:,0]
    data = _ut.interp(CECE[:,0], CECE[:, 1:], ei=None, xo=tt)
    data = _np.hstack((data, ECE[:,1:]))
    return tt, data
    
def genREFsig(frq, tref):
    return _dsp.square(2*_np.pi()*frq*tref, duty=0.5)
    

def extractsig
    
    #2nd order LPF gets converted to a 4th order LPF by filtfilt
    lowpass_n, lowpass_d = _dsp.butter(2, 1.0/(Fs), btype='low')    
    
    for ii in range(nch):
        u_n = _dsp.filtfilt(lowpass_n, lowpass_d, u_t[:, ii])  
    # end for
        

class calHJece(_ut.Struct()):
    def __init__():
        f
        
    # end def
# end class