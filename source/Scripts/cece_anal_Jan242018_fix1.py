#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:37:58 2017

@author: weir
"""

import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
from scipy import signal as _sig
#import scipy.signal.correlate as xcorr

from pybaseutils.fft_analysis import fftanal, butter_lowpass
#from pybaseutils.plt_utils import savefig
from pybaseutils.filter import iirnotch

# datafolder = _os.path.abspath(_os.path.join('~','bin'))
datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

cmPerGHz = 1

#scantitl = 'CECE_jan17_nelow'
# scantitl += '_50to400'
#scantitl += '_400to500'

#freq_ref = 60.0   # [GHz]
#fils = ['CECE.69756','CECE.69757','CECE.69758','CECE.69759','CECE.69760','CECE.69761','CECE.69761','CECE.69762', 'CECE.69777']
#freqs = [13.075,     13.075,      13.095,      13.115,       13.055,      13.035,      13.045,     13.075,      13.080]

freq_ref = 60.0   # [GHz]
fils = ['CECE.69769','CECE.69770','CECE.69771','CECE.69772','CECE.69773','CECE.69777']
freqs = [13.075,     13.075,      13.085,      13.095,      13.105,      13.08]

#fils = ['CECE.69775']
#freqs = [13.080]
freqs = [4.0*freq+8.0 for freq in freqs]

intb = [2e3, 120e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode

#tb = [0.200, 0.280]
#tb = [0.250, 0.300]  # collapse
tb = [0.280, 0.310]  #

Fs = 1.0e6       # [Hz], Sampling frequency
BW = 150e6
Bvid = intb[1]
tint = tb[1]-tb[0]   # seconds
sens = _np.sqrt(2*Bvid/BW/_np.sqrt(2*Bvid*tint))
nfils = len(fils)


f0 = True
# ===== #
if f0:
    #  Make a band rejection / Notch filter for MHD or electronic noise
    f0 = 80.0428e3      # [Hz], frequency to reject
    Q = 30.0         # Quality factor of digital filter
    w0 = f0/(0.5*Fs)   # Normalized frequency
    
    # Design the notch
    #b, a = _sig.iirnotch(w0, Q)  # scipy 0.19.1
    b, a = iirnotch(w0, Q)  # other
    
    # Frequency response
    w, h = _sig.freqz(b,a)
    freq = w*Fs/(2.0*_np.pi)    # Frequency axis
    
    # Plot the response of the filter
    fig, ax = _plt.subplots(2,1, figsize=(8,6), sharex=True)
    ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
    ax[0].set_title('Frequency Response of Notch filter (%4.1f KHz, Q=%i)'%(f0*1e-3, Q))
    ax[0].set_ylabel('Amplitude [dB]')
    xlims = [0, int(1e-3*Fs/2)]
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([-25, 10])
    ax[0].grid()
    ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
    ax[1].set_ylabel('Angle [deg]', color='green')
    ax[1].set_xlabel('Frequency [KHz]')
    ax[1].set_xlim(xlims)
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()
    _plt.show()

fLPF=False
if fLPF:
    fLPF = 2*intb[1]      # [Hz], frequency to reject
    lpf_order = 1       # Order of low pass filter
    w0 = f0/(0.5*Fs)   # Normalized frequency
    
    # Design the LPF
    blpf, alpf = butter_lowpass(fLPF, 0.5*Fs, order=lpf_order)           
    
    # Frequency response
    w, h = _sig.freqz(blpf,alpf)
    freq = w*Fs/(2.0*_np.pi)    # Frequency axis
    
    # Plot the response of the filter
    fig, ax = _plt.subplots(2,1, figsize=(8,6), sharex=True)
    ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
    ax[0].set_title('Frequency Response of Low pass filter (%4.1f KHz order=%i)'%(f0*1e-3, lpf_order))
    ax[0].set_ylabel('Amplitude [dB]')
    xlims = [0, int(1e-3*Fs/2)]
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([-25, 10])
    ax[0].grid()
    ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
    ax[1].set_ylabel('Angle [deg]', color='green')
    ax[1].set_xlabel('Frequency [KHz]')
    ax[1].set_xlim(xlims)
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()
    _plt.show()
    

# ===== #

hfig0, ax0 = _plt.subplots(nfils,1, figsize=(8,6), sharex=True)
hfig1, ax1 = _plt.subplots(nfils,1, figsize=(8,6), sharex=True)
ax0[0].set_title('RF')
ax1[0].set_title('IF')

# ===== #

hfig, ax = _plt.subplots(5,1, figsize=(8,6))
#ax[0].set_xlabel('freq [KHz]')
ax[0].set_ylabel('Cross Power')
ax[1].set_ylabel('Phase')
ax[2].set_ylabel('Coh')
ax[2].set_xlabel('freq [KHz]')
ax[0].get_shared_x_axes().join(ax[0], ax[1], ax[2])

ax[3].set_ylabel('Coh')    
ax[4].set_ylabel('Phase Diff')    
ax[4].set_xlabel('freq [GHz]')
ax[3].get_shared_x_axes().join(ax[3], ax[4])

# ===== #

freqs = _np.asarray(freqs, dtype=_np.float64)

Pxy = _np.zeros( (nfils,), dtype=_np.complex64)
Pxx = _np.zeros( (nfils,), dtype=_np.complex64)
Pyy = _np.zeros( (nfils,), dtype=_np.complex64)
#Cxy = _np.zeros( (nfils,), dtype=_np.complex64)

for ii in range(nfils):
    
    filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
    print(filn)
    tt, tmpRF, tmpIF, _, _ = \
        _np.loadtxt(filn, dtype=_np.float64, unpack=True)

    # plot the signals
    ax0[ii].plot(tt, tmpRF, '-')
    ax0[ii].set_ylabel('%5s'%(fils[ii][5:]))   
    ax1[ii].plot(tt, tmpIF, '-')
    ax1[ii].set_ylabel('%5s'%(fils[ii][5:]))
    if ii == nfils-1:
        ax0[ii].set_xlabel('t [ms]')
        ax1[ii].set_xlabel('t [ms]') 

    tt = 1e-3*tt
    j0 = _np.floor( (tb[0]-tt[0])/(tt[1]-tt[0]))
    j1 = _np.floor( (tb[1]-tt[0])/(tt[1]-tt[0]))        

    if f0:
        # Apply a zero-phase digital filter to both signals
    #    tmpRF = _sig.filtfilt(b, a, tmpRF)  # padding with zeros
    #    tmpIF = _sig.filtfilt(b, a, tmpIF)  # padding with zeros    
        tmpRF = _sig.filtfilt(b, a, tmpRF, method='gust')  # Gustafson's method, no padding
        tmpIF = _sig.filtfilt(b, a, tmpIF, method='gust')  # Gustafson's method, no padding

    if fLPF:
        tmpRF = _sig.filtfilt(blpf, alpf, tmpRF, method='gust')
        tmpIF = _sig.filtfilt(blpf, alpf, tmpIF, method='gust')
    
#    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=5000)      
    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=2000, windowoverlap=0.75)      
#    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=2*int(1e3*tint), windowoverlap=0.5, windowfunction='box')    
#    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=160, windowoverlap=0.0, windowfunction='box')
#    IRfft = fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=int(1e3*tint), windowoverlap=0.0, windowfunction='box')    
#    ircor = xcorr(tmpRF[j0:j1], tmpIF[j0:j1])
    Navr = IRfft.Navr
    ovrlp = IRfft.overlap
    winfn = IRfft.window
    
    # ================ #
    
    freq = IRfft.freq
    i0 = _np.floor( (intb[0]-freq[0])/(freq[1]-freq[0]))
    i1 = _np.floor( (intb[1]-freq[0])/(freq[1]-freq[0]))    
    i0 = int(i0)
    i1 = int(i1)
    ax[0].plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')    
    ax[1].plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')
    ax[2].plot(1e-3*IRfft.freq, _np.sqrt(IRfft.Cxy), '-')
    
    # ================ #
    
    reP = _np.real(IRfft.Pxy)
    imP = _np.imag(IRfft.Pxy)
    
    Pxy[ii] = _np.trapz(reP[i0:i1], x=freq[i0:i1]) \
                + 1j*_np.trapz(imP[i0:i1], x=freq[i0:i1])    
    Pxx[ii] = _np.trapz(IRfft.Pxx[i0:i1], x=freq[i0:i1])
    Pyy[ii] = _np.trapz(IRfft.Pyy[i0:i1], x=freq[i0:i1])
    
    # ================ #
    
# end loop    

# Coherence calculation
Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) ) # mean-squared coherence
CohLim = 1.0/Navr    # mean-squared coherence sensitivity limit
Cxy = _np.sqrt(Cxy)  # coherence
CohLim = _np.sqrt(CohLim)   # coherence sensitivity limit

# Phase calculation
phxy = _np.angle(Pxy)
isort = _np.argsort(_np.abs(freqs-freq_ref))
phxy -= phxy[isort][0]
print("Sensitivity to Te-fluctuations:%4.2f percent \nCoherence significance level: %5.3f"%(100*sens, CohLim))

# =============== #
   
ax[0].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
ax[0].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
ax[1].axvline(x=1e-3*freq[i0], linewidth=2, color='k')
ax[1].axvline(x=1e-3*freq[i1], linewidth=2, color='k')
ax[3].plot(freqs, Cxy, 'o')
ax[3].axhline(y=CohLim, linewidth=2, color='k')
ax[3].text(_np.average(freqs), 0.05, 
          '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
          fontsize=12)
ax[4].plot(freqs, phxy, 'o')
#ax[4].text(_np.average(freqs), 0.05, 
#          '%i to %i GHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])),
#          fontsize=12)

#_plt.figure()
#_plt.xlabel('r [cm]')
#_plt.ylabel('Cross Coherence')
#_plt.plot(_np.abs(freqs-68.0)*cmPerGHz, Cxy, 'o' )

#savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)    
#savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)