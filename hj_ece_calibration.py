#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:49:03 2017

@author: weir
"""

import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
import matplotlib.mlab as _mlb

import scipy.signal as _dsp
import scipy.optimize as _opt

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
    data = _ut.interp(CECE[:, 0], CECE[:, 1:], ei=None, xo=tt)
    data = _np.hstack((data, ECE[:, 1:]))
    return tt, data
    
def mkTTL(refsig):
    TTL = refsig.copy()
    TTL -= _np.mean(refsig)
    TTL[_np.where(TTL>0)] = 3.0
    TTL[_np.where(TTL<0)] = 0.0
    return TTL

class digitallockin(_ut.Struct):
    def __init__(self, tt, sig, frq, fbw, ds_Fs=None, detrend=True, plotit=True):

        self.plotit = plotit
        self.tt = tt
        self.sig = sig
        self.frq = frq
        self.fbw = fbw

        if detrend is not None:
            self.sig = _fft.detrend_mean(self.sig)
        # end if
        
        if ds_Fs is not None:
            self.Fs_old = 1/(tt[1]-tt[0])
            self.Fs_new = ds_Fs            
            self.downsample()
        # end if
    # end def __init__
    
    def downsample(self):
        sig = _fft.downsample(self.sig, self.Fs_old, self.Fs_new, plotit=self.plotit)            
        tt = _np.linspace(self.tt[0], self.tt[-1], _np.size(sig, axis=0))
        sig = sig.reshape((len(tt),))
#        print(_np.shape(self.sig))
#        print(_np.shape(sig))
#        print(_np.shape(tt))
        self.sig = sig
        self.tt = tt
        return self.tt, self.sig
    
    def genREFsig(self, fi, tr):
        return _dsp.square(2*_np.pi*self.frq*tr-fi, duty=0.5)
    
    def returnmaxF(self):
        freq, _, Pxx, _, _, _, _ = \
            _fft.fft_pwelch(self.tt, self.sig, self.sig, [self.tt[1], self.tt[-2]], 
                            useMLAB=True, plotit=self.plotit, detrend_style=1)
#                            _np.asarray([self.tt[1], self.tt[-2]], dtype=_np.float64), 
#        Pxx, freq = _mlb.psd(self.sig, NFFT=2**18, Fs=1./(self.tt[2]-self.tt[1]),
#                             scale_by_freq=True)
        
        ib0 = _np.where(freq>self.frq-self.fbw)[0][0]-1
        ib1 = _np.where(freq>self.frq+self.fbw)[0][0]+1
        ib = _ut.argmax((_np.abs(Pxx[ib0:ib1]).tolist()))+ib0  
        self.frq = freq[ib]
        return self.frq

    def zero_signal(self, fi):
        self.returnmaxF()
        nper = _np.floor((self.tt[-1]-self.tt[0])*self.frq)
        ib = _np.where(self.tt-self.tt[0]>nper/self.frq)[0][0]-1
        return _np.mean(_np.sign(self.genREFsig(fi-_np.pi, self.tt[:ib])) * self.sig[:ib])
        
    def signal_amplitude(self, fi):
        self.returnmaxF()        
        nper = _np.floor((self.tt[-1]-self.tt[0])*self.frq)
        ib = _np.where(self.tt-self.tt[0]>nper/self.frq)[0][0]-1
        return _np.mean(_np.sign(self.genREFsig(fi, self.tt[:ib]))*self.sig[:ib])
        
    def run(self):
        x0 = _np.random.uniform(low=-_np.pi, high=_np.pi, size=1)
        bnds = ((-0.99*_np.pi, _np.pi),)
        tols = 1e-8
        opts = {'maxiter': 1000, 'disp': True}
        res = _opt.minimize(self.zero_signal, x0, bounds=bnds, tol=tols, options=opts)        
        self.res = res
        self.fi = res.x[0]
        return self.signal_amplitude(self.fi), self.frq, self.fi, res

#    def results(self):        
#        return self.signal_amplitude(self.fi), self.frq, self.fi, self.res
#def digitallockin(tt, sig, frq, fbw):

        
def test(nargout=0):

    amp_in = 0.003    
    
    # === # 
    # Noisy test signal
    tt = _np.linspace(150e-3, 400e-3, num=1e6*250e-3)
    sig = _dsp.square(2*_np.pi*91.3432*tt-0.352, duty=0.5)
    sig *= amp_in

    frq = 90.0
    fbw = 10.0        

    # === #
    
    for SN in [0.1, 0.01, 0.001]:
        sigin = sig + SN*_np.random.randn(len(tt))

        diglock = digitallockin(tt, sigin, frq, fbw, plotit=True)
#        diglock = digitallockin(tt, sigin, frq, fbw, ds_Fs=10e3, plotit=True)        
        amp_out, frq_out, fi_out, res = diglock.run()
#        amp_out, frq_out, fi_out, res = digitallockin(tt, sigin, frq, fbw)           
    
        print('-----')
        print(res)        
        print('Amplitude in: %4.4f, Amplitude out: %4.4f'%(amp_in, amp_out))        
        print('Percent difference %2.1f at S/N ratio of %4.4f'%(100*(amp_in-amp_out)/amp_in, SN))
        print('output frequency = %2.1f and phase = %4.4f'%(frq_out, fi_out))
        print('-----')
    # end for
    if nargout==0:
        return 
    elif nargout == 1:
        return 1
    elif nargout == 2:
        return tt, amp_out
    elif nargout == 3:
        return tt, sigin, amp_out
    elif nargout == 4:
        return tt, sigin, amp_out, frq
    elif nargout == 5:
        return tt, sigin, amp_out, frq, fi_out        
# end def

if __name__ == "__main__":
    tt, sig, amp, frq, fi_out = test(5)



