#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:37:58 2017

@author: weir
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

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'Workshop'))
#datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

cmPerGHz = 1
#nwindows = 500
#overlap=0.50
minFreq=0.1e3
scantitl = 'CECE_jan17_fix4'
# scantitl += '_50to400'
#scantitl += '_400to500'

freq_ref = 60.0   # [GHz]
fils = ['CECE.69769','CECE.69770','CECE.69771','CECE.69772','CECE.69773','CECE.69777']
freqs = [13.075,     13.075,      13.085,      13.095,      13.105,      13.08]
freqs = [4.0*freq+8.0 for freq in freqs]

#fils = ['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647']
#freqs = [68.0,       68.3,        68.2,         68.1,       68.15,       68.05]
#
#fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
#freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])

intb = [15e3, 200e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode

tb=[0.15,0.25]
sintest=True
#tb=[0.3,0.39]
#tb = [0.192, 0.370]
f0=False
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
    
    # Plot the response of the filter
#    fig, ax = _plt.subplots(2,1, figsize=(8,6), sharex=True)
#    ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
#    ax[0].set_title('Frequency Response of Notch filter (%4.1f KHz, Q=%i)'%(f0*1e-3, Q))
#    ax[0].set_ylabel('Amplitude [dB]')
#    xlims = [0, int(1e-3*Fs/2)]
#    ax[0].set_xlim(xlims)
#    ax[0].set_ylim([-25, 10])
#    ax[0].grid()
#    ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
#    ax[1].set_ylabel('Angle [deg]', color='green')
#    ax[1].set_xlabel('Frequency [KHz]')
#    ax[1].set_xlim(xlims)
#    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
#    ax[1].set_ylim([-90, 90])
#    ax[1].grid()
#    _plt.show()

fLPF=False
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
    
    # Plot the response of the filter
#    fig, ax = _plt.subplots(2,1, figsize=(8,6), sharex=True)
#    ax[0].plot(1e-3*freq, 20*_np.log10(_np.abs(h)), color='blue')
#    ax[0].set_title('Frequency Response of Low pass filter (%4.1f KHz order=%i)'%(f0*1e-3, lpf_order))
#    ax[0].set_ylabel('Amplitude [dB]')
#    xlims = [0, int(1e-3*Fs/2)]
#    ax[0].set_xlim(xlims)
#    ax[0].set_ylim([-25, 10])
#    ax[0].grid()
#    ax[1].plot(1e-3*freq, _np.unwrap(_np.angle(h))*180.0/_np.pi, color='green')
#    ax[1].set_ylabel('Angle [deg]', color='green')
#    ax[1].set_xlabel('Frequency [KHz]')
#    ax[1].set_xlim(xlims)
#    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
#    ax[1].set_ylim([-90, 90])
#    ax[1].grid()
#    _plt.show()

hfig = _plt.figure()
sub1 = _plt.subplot(4, 1, 1)
sub1.set_xlabel('freq [KHz]')
sub1.set_ylabel('Cross Power [a.u.]')

sub2 = _plt.subplot(4, 1, 2, sharex=sub1)
# sub2.set_ylim((0, 1))
sub2.set_xlabel('freq [KHz]')
sub2.set_ylabel('RMS Coherence')

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
nfils = 1
freqs = _np.asarray(freqs[0:nfils], dtype=_np.float64)

Pxy = _np.zeros( (nfils,), dtype=_np.complex64)
Pxx = _np.zeros( (nfils,), dtype=_np.complex64)
Pyy = _np.zeros( (nfils,), dtype=_np.complex64)
Cxy = _np.zeros( (nfils,), dtype=_np.complex64)
phxy= _np.zeros( (nfils,), dtype=_np.complex64)


MaxFreq=[]
MinFreq=[]
Pxy_avg=0.0
Pxx_avg=0.0
Pyy_avg=0.0
Corr_avg=0.0
Pxy_var=0.0
Pxx_var=0.0
Pyy_var=0.0
Corr_var=0.0
Ex_avg = 0.0   # careful about implicit typing with 0 (int) vs 0.0 (float)
Ey_avg = 0.0
for ii in range(nfils):
    if sintest:
        df=1e3
        ampRF = 1.00
        ampIF = 1.00
        delay_ph=-0.5*_np.pi #time delay in phase shift
        delay_t=delay_ph/(2.0*_np.pi*(df)) #time delay in seconds

        _np.random.seed()

        n_s=1201          
#        n_s=1200
        periods=20.0         
        tt=_np.linspace(0,periods/df,n_s)    
        tb = [tt[0], tt[-10]] #some way or another he does not like tt[-1] in the fftanal
#        n_s=4000001   
        fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))

        tmpRF = ampRF*_np.sin(2.0*_np.pi*(df)*tt)
#        tmpRF += 1.00*ampRF*_np.random.standard_normal( size=(tt.shape[0],) )   # there was an error here
        tmpRF += 10.50*ampRF*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )

        tmpIF = ampIF*_np.sin(2.0*_np.pi*(df)*(tt)+delay_ph)
#        tmpIF += 1.00*ampIF*_np.random.standard_normal( size=(tt.shape[0],) )
#        tmpIF += 1.00*ampIF*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )
        tt_tb=[_np.where(tt<=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
        
        Bvid=0.5e6
        Bif=200e6        
        sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
        sens = _np.sqrt(2*Bvid/Bif/sqrtNs)
    if not sintest:
        
        filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
        print(filn)
        tt, tmpRF, tmpIF = \
            _np.loadtxt(filn, dtype=_np.float64, unpack=True, usecols=(0,1,2))
        tt = 1e-3*tt
        tt_tb=[_np.where(tt==tb[0])[0][0],_np.where(tt==tb[1])[0][0]]
#    [tau,co]=ccf(tmpRF,tmpIF,(len(tt[tt_tb[0]:tt_tb[1]])-1)/(tt[tt_tb[1]]-tt[tt_tb[0]]))
#    _plt.figure()
#    _plt.plot(tau,co)
    
    if fLPF:
        tmpRF = _sig.filtfilt(blpf, alpf, tmpRF.copy())
        tmpIF = _sig.filtfilt(blpf, alpf, tmpIF.copy())
        
    if f0:
        # Apply a zero-phase digital filter to both signals
        tmpRF = _sig.filtfilt(b, a, tmpRF.copy())  # padding with zeros
        tmpIF = _sig.filtfilt(b, a, tmpIF.copy())  # padding with zeros 
       
    if tt[1]-tt[0]!=tt[2]-tt[1]:
        tt2=_np.linspace(tt[0],tt[-1],len(tt),endpoint=True)
        tmpRF=_np.interp(_np.asarray(tt2,dtype=float), tt, tmpRF.copy())
        tmpIF=_np.interp(_np.asarray(tt2,dtype=float), tt, tmpIF.copy())
        tt=tt2
    
    
    j0 = int(_np.floor( 1 + (tb[0]-tt[0])/(tt[1]-tt[0])))
    j1 = int(_np.floor( 1 + (tb[1]-tt[0])/(tt[1]-tt[0])))        
    if j0<=0:        j0 = 0        # end if
    if j1>=len(tt):  j1 = -1       # end if
    
    IRfft = fftanal(tt, tmpRF.copy(), tmpIF.copy(), tbounds=tb, 
                    windowfunction='SFT3M', minFreq=minFreq, plotit=True) 
    #possible windowfunctions: 'SFT3M', 'boxcar', 'Hamming', 'Hanning', 'SFT3F', many others
    #Navr=nwindows, windowoverlap=overlap, windowfunction='box'     

    MaxFreq=_np.append(MaxFreq,IRfft.Fs/2.)
    print('Maximal frequency: '+str(MaxFreq[ii])+' Hz')
    MinFreq=_np.append(MinFreq,2.*IRfft.Fs/IRfft.fftinfo.nwins)
    print('Minimal frequency: '+str(MinFreq[ii])+' Hz')
#    ircor = xcorr(tmpRF[j0:j1], tmpIF[j0:j1])

    # ---------------- #
    
    freq = IRfft.freq
    freq_intb=_np.where((freq>=intb[0])*(freq<=intb[1]))[0]
#    i0 = int(_np.floor( 1 + (intb[0]-freq[0])/(freq[1]-freq[0])))
#    i1 = int(_np.floor( 1 + (intb[1]-freq[0])/(freq[1]-freq[0])))    
#    if i0<=0:          i0 = 0        # end if
#    if i1>=len(freq):  i1 = -1       # end if

                                
    #    sub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(IRfft.Pxy), '-')  
    sub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')    
    sub1.set_ylabel('Cross Power [dB]') 
    sub2.plot(1e-3*IRfft.freq, _np.abs(IRfft.Cxy), '-')
    sub3.plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')
    _plt.close(hfig)
    # ---------------- #
    
#    reP = _np.real(IRfft.Pxy)
#    imP = _np.imag(IRfft.Pxy)
#    
#    Pxy[ii] = _np.trapz(reP[freq_intb], x=freq[freq_intb]) \
#                + 1j*_np.trapz(imP[freq_intb], x=freq[freq_intb])    
    Pxy[ii] = _np.trapz(IRfft.Pxy[freq_intb], x=freq[freq_intb])  # works with complex numbers
    Pxx[ii] = _np.trapz(IRfft.Pxx[freq_intb], x=freq[freq_intb])
    Pyy[ii] = _np.trapz(IRfft.Pyy[freq_intb], x=freq[freq_intb])
    
    # ---------------- #
    
    m_prev = _np.copy(Pxy_avg)
    Pxy_avg += (IRfft.Pxy - Pxy_avg) / (ii+1)
    Pxy_var += (IRfft.Pxy - Pxy_avg) * (IRfft.Pxy - m_prev)        
    
    m_prev = _np.copy(Pxx_avg)    
    Pxx_avg += (IRfft.Pxx - Pxx_avg) / (ii+1)
    Pxx_var += (IRfft.Pxx - Pxx_avg) * (IRfft.Pxx - m_prev)        
    
    m_prev = _np.copy(Pyy_avg)    
    Pyy_avg += (IRfft.Pyy - Pyy_avg) / (ii+1)
    Pyy_var += (IRfft.Pyy - Pyy_avg) * (IRfft.Pyy - m_prev)    

    # Average Cross-correlation of one frequency component
    Ex_tmp = _np.trapz(tmpRF[j0:j1]*tmpRF[j0:j1], x=tt[j0:j1])
    Ey_tmp = _np.trapz(tmpIF[j0:j1]*tmpIF[j0:j1], x=tt[j0:j1])
    Ex_avg += (Ex_tmp.copy() - Ex_avg) / (ii+1)
    Ey_avg += (Ey_tmp.copy() - Ey_avg) / (ii+1)
    
    npts = len(IRfft.freq)
#    xCorr = _np.sqrt(npts)*_np.fft.ifft(IRfft.Cxy, n=npts)  # you need to inverse FFT shift here, and it should be Pxy    
#    xCorr = _np.sqrt(npts)*_np.fft.ifft(IRfft.Cxy, n=npts)
#    xCorr = _np.fft.fftshift(xCorr)
    xCorr = _np.sqrt(npts)*_np.fft.ifft(_np.fft.ifftshift(IRfft.Pxy), n=npts)  # you need to inverse FFT shift here
    xCorr = _np.fft.fftshift(xCorr) / _np.sqrt( Ex_tmp*Ey_tmp )
    
    m_prev = _np.copy(Corr_avg)            
    Corr_avg += (xCorr - Corr_avg) / (ii+1)
    Corr_var += (xCorr - Corr_avg) * (xCorr - m_prev) 
    
# end loop    
Cxy2 = _np.abs( Pxy*Pxy.conjugate()) / (_np.abs(Pxx)*_np.abs(Pyy) )  # Mean-squared coherence (integrated)
Cxy = _np.sqrt( Cxy2 )      # RMS coherence (integrated)
phxy = _np.arctan2(_np.imag(Pxy), _np.real(Pxy))   # cross-phase (integrated)
phxy[_np.where(phxy<2.7)] += _np.pi                
phxy[_np.where(phxy>2.7)] -= _np.pi    

Coh = Pxy / _np.sqrt(_np.abs(Pxx)*_np.abs(Pyy) )  # Complex Coherence (integrated)
Coh_avg = Pxy_avg / _np.sqrt(_np.abs(Pxx_avg)*_np.abs(Pyy_avg))    # complex coherence spectra
Coh_var = ((1.0-Coh_avg*_np.conj(Coh_avg))/_np.sqrt(2*IRfft.Navr))**2.0 # spectra

# --------------- #
   
#sub1.axvline(x=1e-3*freq[freq_intb[0]], linewidth=2, color='k')
#sub1.axvline(x=1e-3*freq[freq_intb[-1]], linewidth=2, color='k')
#sub2.axvline(x=1e-3*freq[freq_intb[0]], linewidth=2, color='k')
#sub2.axvline(x=1e-3*freq[freq_intb[-1]], linewidth=2, color='k')
#sub2.axhline(y=1./IRfft.Navr, linewidth=2, color='k')
##sub3.axvline(x=1e-3*freq[freq_intb[0]], linewidth=2, color='k')
##sub3.axvline(x=1e-3*freq[freq_intb[-1]], linewidth=2, color='k')

sub4.plot(freqs, _np.sqrt(Cxy), 'o')
ylims = sub4.get_ylim()
sub4.set_ylim( 0, ylims[1] ) 
sub4b.plot(freqs, phxy, 'rx')

#sub4.text(_np.average(freqs), 0.05, 
#          '%i to %i GHz'%(int(1e-3*freq[freq_intb[0]]),int(1e-3*freq[freq_intb[-1]])),
#          fontsize=12)

#_plt.figure()
#_plt.xlabel('r [cm]')
#_plt.ylabel('Coherence')
#_plt.plot(_np.abs(freqs-68.0)*cmPerGHz, Cxy, 'o' )

#savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)    
#savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)




''' Average cross-correlation from the inverse FFT'''  
npts = len(IRfft.freq)
#lags = _np.asarray(_np.arange(-npts//2, npts//2), dtype=_np.float64) 
#lags /= float(IRfft.Fs)
#lags *= -1.0
lags = (_np.asarray(range(1, IRfft.nwins+1), dtype=int)-IRfft.Nnyquist)/IRfft.Fs

CorrAvg = _np.sqrt(npts)*_np.fft.fftshift(_np.fft.ifft(_np.fft.ifftshift(Pxy_avg), n=npts))
CorrAvg /= _np.sqrt( Ex_avg*Ey_avg )
CorrVar = npts*_np.fft.fftshift(_np.fft.ifft(_np.fft.ifftshift(Pxy_var), n=npts))
CorrVar /= Ex_avg*Ey_avg

#CorrAvg = _np.sqrt(npts)*_np.fft.ifft(Cxy_avg, n=npts)
#CorrVar = _np.sqrt(npts)*_np.fft.ifft(Cxy_var, n=npts)
#CorrAvg = _np.fft.fftshift(CorrAvg)
#CorrVar = _np.fft.fftshift(CorrVar)

#clrs = ['b', 'g', 'm', 'r']
#clrs = clrs*8
#corr_title = scantitl+' Correlation'    
#_hCorr = _plt.figure()    
#_aCorr = _hCorr.gca()
## _aCxyRF.set_ylim((0, 1))
#_aCorr.set_xlabel('lags [us]')
#_aCorr.set_ylabel(r'$\rho_{x,y}$')
#_aCorr.set_title(scantitl)  
##_plt.figure()
#_hCorr.sca(_aCorr)
#_aCorr.plot(1e6*lags, _np.abs(CorrAvg), '-', lw=2, color=clrs[1] )
#_aCorr.fill_between(1e6*lags, 
#                   (_np.abs(CorrAvg)-_np.sqrt(_np.abs(CorrVar))), 
#                   (_np.abs(CorrAvg)+_np.sqrt(_np.abs(CorrVar))), 
#                   facecolor=clrs[1], alpha=0.5)
#
#_hCorr = _plt.figure()    
#_aCorr = _hCorr.gca()
## _aCxyRF.set_ylim((0, 1))
#_aCorr.set_xlabel('lags [us]')
#_aCorr.set_ylabel(r'$\rho_{x,y}$')
#_aCorr.set_title(scantitl)  
##_plt.figure()
#_hCorr.sca(_aCorr)
#_aCorr.plot(1e6*lags, _np.abs(Corr_avg), '-', lw=2, color=clrs[1] )
#_aCorr.fill_between(1e6*lags, 
#                   (_np.abs(Corr_avg)-_np.sqrt(_np.abs(Corr_var))), 
#                   (_np.abs(Corr_avg)+_np.sqrt(_np.abs(Corr_var))), 
#                   facecolor=clrs[1], alpha=0.5)

#_plt.figure()
#_plt.plot(tt,tmpRF)
#_plt.plot(tt,tmpIF)

    