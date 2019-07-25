#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:37:58 2017

@author: weir
"""

import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
#import scipy.signal.correlate as xcorr

from pybaseutils.fft_analysis import fftanal
from pybaseutils.plt_utils import savefig

datafolder = _os.path.abspath(_os.path.join('..', 'Workshop'))
#datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

cmPerGHz = 1

scantitl = 'CECE_jan17_fix4'
# scantitl += '_50to400'
#scantitl += '_400to500'


fils = ['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647']
freqs = [68.0,       68.3,        68.2,         68.1,       68.15,       68.05]

fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])

intb = [15e3, 500e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode

tb = [0.192, 0.370]

hfig = _plt.figure()
sub1 = _plt.subplot(4, 1, 1)
sub1.set_xlabel('freq [KHz]')
sub1.set_ylabel('Cross Power [a.u.]')

sub2 = _plt.subplot(4, 1, 2, sharex=sub1)
# sub2.set_ylim((0, 1))
sub2.set_xlabel('freq [KHz]')
sub2.set_ylabel('Coherence')

sub3 = _plt.subplot(4, 1, 3, sharex=sub1)
# sub3.set_ylim((0, 1))
sub3.set_xlabel('freq [KHz]')
sub3.set_ylabel('Phase [rad]')

sub4 = _plt.subplot(4, 1, 4)
# sub4.set_ylim((0, 1))
sub4.set_xlabel('freq [GHz]')
sub4.set_ylabel('Coherence Length')    
sub4b = sub4.twinx()
sub4b.set_ylabel('Phase [rad]', color='r')

nfils = len(fils)
freqs = _np.asarray(freqs, dtype=_np.float64)

Pxy = _np.zeros( (nfils,), dtype=_np.complex64)
Pxx = _np.zeros( (nfils,), dtype=_np.complex64)
Pyy = _np.zeros( (nfils,), dtype=_np.complex64)
Cxy = _np.zeros( (nfils,), dtype=_np.complex64)
phxy= _np.zeros( (nfils,), dtype=_np.complex64)

for ii in range(nfils):
    
    filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
    print(filn)
    tt, tmpRF, tmpIF = \
        _np.loadtxt(filn, dtype=_np.float64, unpack=True)
    
    tt = 1e-3*tt
    j0 = int(_np.floor( 1 + (tb[0]-tt[0])/(tt[1]-tt[0])))
    j1 = int(_np.floor( 1 + (tb[1]-tt[0])/(tt[1]-tt[0])))        
    if j0<=0:        j0 = 0        # end if
    if j1>=len(tt):  j1 = -1       # end if
    
    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=2000)      

#    ircor = xcorr(tmpRF[j0:j1], tmpIF[j0:j1])

    # ---------------- #
    
    freq = IRfft.freq
    i0 = int(_np.floor( 1 + (intb[0]-freq[0])/(freq[1]-freq[0])))
    i1 = int(_np.floor( 1 + (intb[1]-freq[0])/(freq[1]-freq[0])))    
    if i0<=0:          i0 = 0        # end if
    if i1>=len(freq):  i1 = -1       # end if
    
#    sub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')    
#    sub1.set_ylabel('Cross Power [dB]')

    sub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(IRfft.Pxy), '-')        
    sub2.plot(1e-3*IRfft.freq, _np.sqrt(IRfft.Cxy), '-')
    sub3.plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')
    
    # ---------------- #
    
    reP = _np.real(IRfft.Pxy)
    imP = _np.imag(IRfft.Pxy)
    
    Pxy[ii] = _np.trapz(reP[i0:i1+1], x=freq[i0:i1+1]) \
                + 1j*_np.trapz(imP[i0:i1+1], x=freq[i0:i1+1])    
    Pxx[ii] = _np.trapz(IRfft.Pxx[i0:i1+1], x=freq[i0:i1+1])
    Pyy[ii] = _np.trapz(IRfft.Pyy[i0:i1+1], x=freq[i0:i1+1])
    
    # ---------------- #
    
# end loop    
Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) )
Cxy = _np.sqrt( Cxy )
phxy = _np.arctan2(_np.imag(Pxy), _np.real(Pxy))
phxy[_np.where(phxy<2.7)] += _np.pi    
phxy[_np.where(phxy>2.7)] -= _np.pi    

# --------------- #
   
sub1.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
sub1.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
sub2.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
sub2.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
sub3.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
sub3.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
sub4.plot(freqs, _np.sqrt(Cxy), 'o')
ylims = sub4.get_ylim()
sub4.set_ylim( 0, ylims[1] ) 
sub4b.plot(freqs, phxy, 'rx')
sub4.text(_np.average(freqs), 0.05, 
          '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
          fontsize=12)

#_plt.figure()
#_plt.xlabel('r [cm]')
#_plt.ylabel('Coherence')
#_plt.plot(_np.abs(freqs-68.0)*cmPerGHz, Cxy, 'o' )

#savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)    
#savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)