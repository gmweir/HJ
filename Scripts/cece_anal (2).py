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

from FFT.fft_analysis import fftanal
from pybaseutils.plt_utils import savefig

# datafolder = _os.path.abspath(_os.path.join('~','bin'))
# datafolder = _os.path.join('/homea','weir','bin')
datafolder = _os.path.join('D:/','OneDrive - UW-Madison','HJ','Data')
print(datafolder)

cmPerGHz = 1

#scantitl = 'CECE_jan17_nelow'
# scantitl += '_50to400'
#scantitl += '_400to500'


fils = ['CECE.65625','CECE.65626','CECE.65627','CECE.65628','CECE.65629']
freqs = [68.3,       68.0,        68.2,        68.1,         68.1]

fils.extend(['CECE.65630','CECE.65631','CECE.65632','CECE.65633','CECE.65634'])
freqs.extend([68.15,      68.05,       67.95,        67.90,      68.125])

# intb = [15e3, 500e3]  # original
intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode

tb = [0.192, 0.370]

hfig = _plt.figure()
sub1 = _plt.subplot(3, 1, 1)
sub1.set_xlabel('freq [KHz]')
sub1.set_ylabel('Cross Power')

sub2 = _plt.subplot(3, 1, 2)
# sub2.set_ylim((0, 1))
sub2.set_xlabel('freq [KHz]')
sub2.set_ylabel('Cross Coherence')

sub3 = _plt.subplot(3, 1, 3)
# sub3.set_ylim((0, 1))
sub3.set_xlabel('freq [GHz]')
sub3.set_ylabel('Coherence Length')

nfils = len(fils)
freqs = _np.asarray(freqs, dtype=_np.float64)

Pxy = _np.zeros( (nfils,), dtype=_np.complex64)
Pxx = _np.zeros( (nfils,), dtype=_np.complex64)
Pyy = _np.zeros( (nfils,), dtype=_np.complex64)
Cxy = _np.zeros( (nfils,), dtype=_np.complex64)

for ii in range(nfils):

    filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
    print(filn)
    # tt, tmpRF, tmpIF = \
        # _np.loadtxt(filn, dtype=_np.float64, unpack=True)
    tt, tmpRF, tmpIF = \
        _np.loadtxt(filn, dtype=_np.float64, usecols=(0,1,2), unpack=True)

    tt = 1e-3*tt
    j0 = _np.floor( (tb[0]-tt[0])/(tt[1]-tt[0]))
    j1 = _np.floor( (tb[1]-tt[0])/(tt[1]-tt[0]))


    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=5000)
    IRfft.fftpwelch()

#    ircor = xcorr(tmpRF[j0:j1], tmpIF[j0:j1])

    # ---------------- #

    freq = IRfft.freq
    i0 = int(_np.floor( (intb[0]-freq[0])/(freq[1]-freq[0])))
    i1 = int(_np.floor( (intb[1]-freq[0])/(freq[1]-freq[0])))

    sub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')
    sub2.plot(1e-3*IRfft.freq, IRfft.Cxy, '-')

    # ---------------- #

    reP = _np.real(IRfft.Pxy)
    imP = _np.imag(IRfft.Pxy)

    Pxy[ii] = _np.trapz(reP[i0:i1], x=freq[i0:i1]) \
                + 1j*_np.trapz(imP[i0:i1], x=freq[i0:i1])
    Pxx[ii] = _np.trapz(IRfft.Pxx[i0:i1], x=freq[i0:i1])
    Pyy[ii] = _np.trapz(IRfft.Pyy[i0:i1], x=freq[i0:i1])

    # ---------------- #

# end loop
Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) )

# --------------- #

sub1.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
sub1.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
sub2.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
sub2.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
sub3.plot(freqs, Cxy, 'o')
sub3.text(_np.average(freqs), 0.05,
          '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
          fontsize=12)

#_plt.figure()
#_plt.xlabel('r [cm]')
#_plt.ylabel('Cross Coherence')
#_plt.plot(_np.abs(freqs-68.0)*cmPerGHz, Cxy, 'o' )

#savefig(_os.path.join(datafolder,scantitl), ext='png', close=False,
#        verbose=True, dotsperinch = 300, transparency = True)
#savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False,
#        verbose=True, dotsperinch = 300, transparency = True)