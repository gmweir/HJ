# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:41:32 2019

@author: s146959
"""

# ========================================================================== #
# ========================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
#import scipy.signal.correlate as xcorr
from scipy import signal as _sig
from FFT.fft_analysis import fftanal, ccf
from pybaseutils.plt_utils import savefig
from FFT.fft_analysis import butter_lowpass
from FFT.notch_filter import iirnotch

_plt.close("all")

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'Workshop'))
#datafolder = _os.path.join('/homea','weir','bin')
#print(datafolder)

#data options
minFreq=1e3
intb = [15e3, 200e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode
tb=[0.15,0.25]
#filter options
f0=False
fLPF=False
#window options
Navr=10
windowoverlap=0.5

if f0:
    Fs = 1e6
    #  Make a band rejection / Notch filter for MHD or electronic noise
    f0 = 80.48e3      # [Hz], frequency to reject
    Q = 20.0         # Quality factor of digital filter
    w0 = f0/(0.5*Fs)   # Normalized frequency
    
    # Design the notch
    #b, a = _sig.iirnotch(w0, Q)  # scipy 0.19.1
    b, a = iirnotch(w0, Q)  # other
    
    # Frequency response
    w, h = _sig.freqz(b,a)
    freq = w*Fs/(2.0*_np.pi)    # Frequency axis
    
if fLPF:
    fLPF = 2*intb[1]      # [Hz], frequency to reject
    Bvid = fLPF
    lpf_order = 1       # Order of low pass filter
    w0 = f0/(0.5*Fs)   # Normalized frequency
    
    # Design the LPF
    blpf, alpf = butter_lowpass(fLPF, 0.5*Fs, order=lpf_order)           
    
    # Frequency response
    w, h = _sig.freqz(blpf,alpf)
    freq = w*Fs/(2.0*_np.pi)    # Frequency axis

df=15e3 #frequency of sine wave
ampRF = 0.10
ampRFN= 1.0
ampIF = 0.10
ampIFN= 1.0
delay_ph=-0.*_np.pi #time delay in phase shift
delay_t=delay_ph/(2.0*_np.pi*(df)) #time delay in seconds
_np.random.seed()

n_s=2201
periods=20.0         
tt=_np.linspace(0,periods/df,n_s)    
tb = [tt[0], tt[-1]] #some way or another he does not like tt[-1] in the fftanal
fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))

tmpRF = ampRF*_np.sin(2.0*_np.pi*(df)*tt)
#tmpRF += 1.00*ampRF*_np.random.standard_normal( size=(tt.shape[0],) )   # there was an error here
tmpRF += ampRFN*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )

tmpIF = ampIF*_np.sin(2.0*_np.pi*(df)*(tt)+delay_ph)
#tmpIF += 1.00*ampIF*_np.random.standard_normal( size=(tt.shape[0],) )
tmpIF += ampIFN*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )

tt_tb=[_np.where(tt<=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
SignalTime=tt[tt_tb][1]-tt[tt_tb][0]

_plt.figure()
_plt.plot(tt, tmpRF)
_plt.plot(tt, tmpIF)
_plt.title('raw data')
       
#for multiple windows
#calculate window parameters
nsig=len(tmpRF)
nwins=int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + windowoverlap)))
noverlap=int(nwins*windowoverlap)
ist=_np.arange(Navr)*(nwins-noverlap)   #Not actually using 1000 windows due to side effects?

MaxFreq=fs/2.0
print('Maximal frequency: '+str(MaxFreq)+' Hz')
MinFreq=2.0*fs/nwins #2*fr
print('Minimal frequency: '+str(MinFreq)+' Hz') 

Pxxd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)
Pyyd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)
Pxyd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)

Xfft = _np.zeros((Navr, nwins), dtype=_np.complex128)
Yfft = _np.zeros((Navr, nwins), dtype=_np.complex128)

for i in range(Navr-1): #Not actually using 1000 windows due to side effects?
    istart=ist[i]
    iend=ist[i]+nwins
    
    xtemp=tmpRF[istart:iend+1]
    ytemp=tmpIF[istart:iend+1]
    win=_np.hanning(len(xtemp))
    xtemp=xtemp*win
    ytemp=ytemp*win
    
        
    Xfft[i,:nwins]=_np.fft.fft(xtemp,nwins,axis=0)
    Yfft[i,:nwins]=_np.fft.fft(ytemp,nwins,axis=0)
    
fr=1/(tt[iend]-tt[istart])
#Calculate Power spectral denisty and Cross power spectral density per segment
Pxxd_seg[:Navr,:nwins]=(Xfft*_np.conj(Xfft))
Pyyd_seg[:Navr,:nwins]=(Yfft*_np.conj(Yfft))
Pxyd_seg[:Navr,:nwins]=(Yfft*_np.conj(Xfft))

freq = _np.fft.fftfreq(nwins, 1.0/fs)
Nnyquist = nwins//2

#take one sided frequency band and double the energy as it was split over 
#total frequency band
freq = freq[:Nnyquist]  # [Hz]
Pxx_seg = Pxxd_seg[:, :Nnyquist]
Pyy_seg = Pyyd_seg[:, :Nnyquist]
Pxy_seg = Pxyd_seg[:, :Nnyquist]

Pxx_seg[:, 1:-1] = 2*Pxx_seg[:, 1:-1]  # [V^2/Hz],
Pyy_seg[:, 1:-1] = 2*Pyy_seg[:, 1:-1]  # [V^2/Hz],
Pxy_seg[:, 1:-1] = 2*Pxy_seg[:, 1:-1]  # [V^2/Hz],
if nwins%2:  # Odd
    Pxx_seg[:, -1] = 2*Pxx_seg[:, -1]
    Pyy_seg[:, -1] = 2*Pyy_seg[:, -1]
    Pxy_seg[:, -1] = 2*Pxy_seg[:, -1]

#Normalise to RMS by removing gain
S1=_np.sum(win)
Pxx_seg = Pxx_seg/(S1**2)
Pyy_seg = Pyy_seg/(S1**2)
Pxy_seg = Pxy_seg/(S1**2)

#Normalise to true value
S2=_np.sum(win**2)
Pxx_seg = Pxx_seg/(fs*S2/S1**2)
Pyy_seg = Pyy_seg/(fs*S2/S1**2)
Pxy_seg = Pxy_seg/(fs*S2/S1**2)

#Average the different windows
Pxx=_np.mean(Pxx_seg, axis=0)
Pyy=_np.mean(Pyy_seg, axis=0)
Pxy=_np.mean(Pxy_seg, axis=0)        

_plt.figure()
_plt.plot(freq/1000,Pxy)
_plt.xlabel('frequency [kHz]')
_plt.title('Pxy')

#calculate cross coherence, background mean and uncertainty
cc=Pxy/_np.sqrt(Pxx*Pyy)
ccbg=_np.mean(cc[-int(_np.ceil(len(cc)/5)):])
sigmacc=_np.sqrt((1-_np.abs(cc)**2)**2/(2*Navr))

_plt.figure()
_plt.plot(freq/1000,cc)
_plt.plot(freq/1000,_np.real(cc-ccbg))
_plt.plot(freq/1000,2*sigmacc,'--',color='red')
_plt.ylabel(r'Re($\gamma$-$\gamma_{bg}$)')
_plt.xlabel('frequency [kHz]')
_plt.title('Real part of CC and background subtracted CC')
_plt.axvline(MinFreq/1000, color='k')

f1=0e3
f2=20e3
integrand=(_np.real(cc-ccbg)/(1-_np.real(cc-ccbg)))[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]
integralfreqs=freq[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]
integral=_np.trapz(integrand,integralfreqs)
Bvid=0.5e6
Bif=200e6        
sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
sens = _np.sqrt(2*Bvid/Bif/sqrtNs) 
Tfluct=_np.sqrt(2*integral/Bif)
print('Tfluct/T= '+str(Tfluct))

    
 
