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
minFreq=10e3
intb = [15e3, 25e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode
#window options
Navr=100
windowoverlap=0.5

#input options
df=20e3 #frequency of sine wave
ampRF = 0.10
ampRFN= 1.0
ampIF = 0.10
ampIFN= 1.0
delay_ph=-0.*_np.pi #time delay in phase shift
delay_t=delay_ph/(2.0*_np.pi*(df)) #time delay in seconds
_np.random.seed()

n_s=20001
periods=1000.0     
tt=_np.linspace(0,periods/df,n_s)    
tb = [tt[0], tt[-1]] #some way or another he does not like tt[-1] in the fftanal
fs=1/(((tt[len(tt)-1]-tt[0])/(len(tt)-1)))

tmpRF = ampRF*_np.sin(2.0*_np.pi*(df)*tt)
#tmpRF += 1.00*ampRF*_np.random.standard_normal( size=(tt.shape[0],) )   # there was an error here
tmpRF += ampRFN*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )

tmpIF = ampIF*_np.sin(2.0*_np.pi*(df)*(tt)+delay_ph)
#tmpIF += 1.00*ampIF*_np.random.standard_normal( size=(tt.shape[0],) )
tmpIF += ampIFN*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )


_plt.figure()
_plt.plot(tt, tmpRF)
_plt.plot(tt, tmpIF)
_plt.title('raw data')
       
#calculate window parameters
nsig=len(tmpRF)
nwins=int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + windowoverlap)))
noverlap=int(_np.ceil(nwins*windowoverlap))
ist=_np.arange(Navr)*(nwins-noverlap)   #Not actually using 1000 windows due to side effects?

#calculate maximum and minimum discernible frequencies
MaxFreq=fs/2.0
MinFreq=2.0*fs/nwins #2*fr
print('Maximal frequency: '+str(MaxFreq)+' Hz')
print('Minimal frequency: '+str(MinFreq)+' Hz') 

#create empty matrices to fill up
Pxxd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)
Pyyd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)
Pxyd_seg = _np.zeros((Navr, nwins), dtype=_np.complex128)

Xfft = _np.zeros((Navr, nwins), dtype=_np.complex128)
Yfft = _np.zeros((Navr, nwins), dtype=_np.complex128)

for i in range(Navr):
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
_plt.plot(freq/1000,_np.abs(Pxy))
_plt.xlabel('frequency [kHz]')
_plt.title('Pxy')

f1=max((MinFreq, intb[0]))
f2=min((MaxFreq, intb[1]))
if1 = _np.where(freq>=f1)[0]
if2 = _np.where(freq>=f2)[0]
if1 = 0 if len(if1) == 0 else if1[0]
if2 = len(freq) if len(if2) == 0 else if2[0]
ifreqs = _np.asarray(range(if1, if2), dtype=int)
integralfreqs=freq[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]

cc2i = (_np.trapz(Pxy[ifreqs],integralfreqs)
    / _np.sqrt(_np.trapz(_np.abs(Pxx[ifreqs]),integralfreqs)*_np.trapz(_np.abs(Pyy[ifreqs]),integralfreqs)))

_plt.figure()
_ax1 = _plt.subplot(3,1,1)
_ax2 = _plt.subplot(3,1,2, sharex=_ax1)
_ax3 = _plt.subplot(3,1,3, sharex=_ax1)
_ax1.plot(1e-3*freq,10*_np.log10(_np.abs(Pxy)))
_ax1.set_ylabel(r'P$_{xy}$ [dB/Hz]')
_ax2.plot(1e-3*freq,10*_np.log10(_np.abs(Pxx)))
_ax2.set_ylabel(r'P$_{xx}$ [dB/Hz]')
_ax3.plot(1e-3*freq,10*_np.log10(_np.abs(Pyy)))
_ax3.set_ylabel(r'P$_{yy}$ [dB/Hz]')
_ax3.set_xlabel('f [KHz]')
_ylims = _ax1.get_ylim()
_ax1.axvline(x=1e-3*freq[ifreqs[0]], ymin=_ylims[0],
         ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[0]]/_ylims[1],
         linewidth=0.5, linestyle='--', color='black')
_ax1.axvline(x=1e-3*freq[ifreqs[-1]], ymin=_ylims[0],
         ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[-1]]/_ylims[1],
         linewidth=0.5, linestyle='--', color='black')

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

integrand=(_np.real(cc-ccbg)/(1-_np.real(cc-ccbg)))[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]
integral=_np.trapz(integrand,integralfreqs)
Bvid=0.5e6
Bif=200e6        
sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
sens = _np.sqrt(2*Bvid/Bif/sqrtNs) 
Tfluct=_np.sqrt(2*integral/Bif)
sigmaTfluct=_np.sqrt(_np.sum((sigmacc*fr)**2))/(Bif*Tfluct)
msg = u'Tfluct/T=%2.3f\u00B1%2.3f%%'%(100*Tfluct, 100*sigmaTfluct)
print(msg)


#Perform the same analysis using fftanal    
sig_anal = fftanal(tt.copy(), tmpRF.copy(), tmpIF.copy(), windowfunction='hanning',
                  onesided=True, windowoverlap=windowoverlap, Navr=Navr, plotit=False) 
sig_anal.fftpwelch()

_nwins = sig_anal.nwins
_fs = sig_anal.Fs
_fr = fs/float(nwins)
_Navr = sig_anal.Navr
_freq = sig_anal.freq.copy()
_Pxy = sig_anal.Pxy.copy()
_Pxx = sig_anal.Pxx.copy()
_Pyy = sig_anal.Pyy.copy()

_MaxFreq=fs/2.
_MinFreq=2.0*fs/nwins #2*fr
print('Maximal frequency: '+str(_MaxFreq)+' Hz')
print('Minimal frequency: '+str(_MinFreq)+' Hz')

_plt.figure()
_plt.plot(_freq/1000,_Pxy)
_plt.xlabel('frequency [kHz]')
_plt.title('Pxy fftanal')


_siglags = sig_anal.fftinfo.lags.copy()
_sigcorrcoef = sig_anal.fftinfo.corrcoef.copy()

# integration frequencies
_f1=max((_MinFreq, intb[0]))
_f2=min((_MaxFreq, intb[1]))
_if1 = _np.where(_freq>=_f1)[0]
_if2 = _np.where(_freq>=_f2)[0]
_if1 = 0 if len(_if1) == 0 else _if1[0]
_if2 = len(freq) if len(_if2) == 0 else _if2[0]
_ifreqs = _np.asarray(range(_if1, _if2), dtype=int)
_integralfreqs=_freq[_np.where(_freq>=_f1)[0][0]:_np.where(_freq>=_f2)[0][0]]

_cc2i = (_np.trapz(_Pxy[_ifreqs],_integralfreqs)
    / _np.sqrt(_np.trapz(_np.abs(_Pxx[_ifreqs]),_integralfreqs)*_np.trapz(_np.abs(_Pyy[_ifreqs]),_integralfreqs)))

_plt.figure()
_ax1 = _plt.subplot(3,1,1)
_ax2 = _plt.subplot(3,1,2, sharex=_ax1)
_ax3 = _plt.subplot(3,1,3, sharex=_ax1)
_ax1.plot(1e-3*_freq,10*_np.log10(_np.abs(_Pxy)))
_ax1.set_ylabel(r'P$_{xy}$ [dB/Hz]')
_ax2.plot(1e-3*_freq,10*_np.log10(_np.abs(_Pxx)))
_ax2.set_ylabel(r'P$_{xx}$ [dB/Hz]')
_ax3.plot(1e-3*_freq,10*_np.log10(_np.abs(_Pyy)))
_ax3.set_ylabel(r'P$_{yy}$ [dB/Hz]')
_ax3.set_xlabel('f [KHz]')
_ylims = _ax1.get_ylim()
_ax1.axvline(x=1e-3*_freq[_ifreqs[0]], ymin=_ylims[0],
         ymax=10*_np.log10(_np.abs(_Pxy))[_ifreqs[0]]/_ylims[1],
         linewidth=0.5, linestyle='--', color='black')
_ax1.axvline(x=1e-3*_freq[_ifreqs[-1]], ymin=_ylims[0],
         ymax=10*_np.log10(_np.abs(_Pxy))[_ifreqs[-1]]/_ylims[1],
         linewidth=0.5, linestyle='--', color='black')

_cc=_Pxy/_np.sqrt(_Pxx*_Pyy)
_ccbg=_np.mean(_cc[-int(_np.ceil(len(_cc)/5)):])
_sigmacc=_np.sqrt((1-_np.abs(_cc)**2)**2/(2*_Navr))

_plt.figure()
_plt.plot(_freq/1000,_cc)
_plt.plot(_freq/1000,_np.real(_cc-_ccbg))
_plt.plot(_freq/1000,2*_sigmacc,'--',color='red')
_plt.ylabel(r'Re($\gamma$-$\gamma_{bg}$)')
_plt.xlabel('frequency [kHz]')
_plt.title('Real part of CC and background subtracted CC fftanal')
_plt.axvline(_MinFreq/1000, color='k')

_integrand=(_np.real(_cc-_ccbg)/(1-_np.real(_cc-_ccbg)))[_np.where(_freq>=_f1)[0][0]:_np.where(_freq>=_f2)[0][0]]
_integral=_np.trapz(_integrand,_integralfreqs)
_Bvid=0.5e6
_Bif=200e6        
_sqrtNs = _np.sqrt(2*_Bvid*(tb[-1]-tb[0]))
_sens = _np.sqrt(2*_Bvid/Bif/_sqrtNs)
_Tfluct=_np.sqrt(2*_integral/_Bif)
_sigmaTfluct=_np.sqrt(_np.sum((_sigmacc*_fr)**2))/(_Bif*_Tfluct)
_msg = u'Tfluct/T=%2.3f\u00B1%2.3f%%'%(100*_Tfluct, 100*_sigmaTfluct)
print(_msg)

_plt.figure('RMS Coherence')
_plt.plot(df*1e-3, _np.abs(cc2i), 'o' )
_plt.axhline(y=1.0/_np.sqrt(Navr), linestyle='--')
_plt.xlabel('freq [kHz]')
_plt.ylabel(r'RMS Coherence')

_plt.figure('RMS Coherence')
_plt.plot(df*1e-3, _np.abs(_cc2i), 'o' )
_plt.axhline(1.0/_np.sqrt(_Navr), linestyle='--')
_plt.xlabel('freq [kHz]')
_plt.ylabel(r'RMS Coherence')

_plt.figure('Tfluct')
_plt.errorbar(df*1e-3, Tfluct, yerr=sigmaTfluct, fmt='o-',capsize=5)
_plt.axhline(sens, linestyle='--')
_plt.xlabel('freq [kHz]')
_plt.ylabel(r'$\tilde{T}_e/<T_e>$')

_plt.figure('Tfluct')
_plt.errorbar(df*1e-3, _Tfluct, yerr=_sigmaTfluct, fmt='o',capsize=5)
_plt.axhline(_sens, linestyle='--')
_plt.xlabel('freq [kHz]')
_plt.ylabel(r'$\tilde{T}_e/<T_e>$')

_plt.figure()
_plt.plot(_freq/1000,_np.real(cc-ccbg))
_plt.plot(_freq/1000,_np.real(_cc-_ccbg))
_plt.plot(_freq/1000,2*_sigmacc,'--',color='red')
_plt.ylabel(r'Re($\gamma$-$\gamma_{bg}$)')
_plt.xlabel('frequency [kHz]')
_plt.title('Homemade and fftanal complex coherance')
_plt.axvline(_MinFreq/1000, color='k')

_plt.figure()
_plt.plot(_freq/1000,_np.real(cc-ccbg)-_np.real(_cc-_ccbg))
_plt.ylabel(r'Re($\gamma$-$\gamma_{bg}$)')
_plt.xlabel('frequency [kHz]')
_plt.title('Difference homemade and fftanal complex coherance')
_plt.axvline(_MinFreq/1000, color='k')