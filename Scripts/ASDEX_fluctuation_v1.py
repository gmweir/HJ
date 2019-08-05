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

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'Workshop'))
#datafolder = _os.path.join('/homea','weir','bin')
#print(datafolder)

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

fils.extend (['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647'])
freqs.extend ([68.0,       68.3,        68.2,         68.1,       68.15,       68.05])
#
fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])
#
#fils=['CECE.65644']
#freqs=[68.2]


intb = [15e3, 200e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
# intb = [400e3, 465e3]  # high frequency mode

tb=[0.15,0.25]
sintest=False
#tb=[0.3,0.39]
#tb = [0.192, 0.370]
f0=True
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
    

fLPF=True
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

nfils = len(fils)
nfils = 5
freqs = _np.asarray(freqs[0:nfils], dtype=_np.float64)

for ii in range(nfils):
    if sintest:
        df=1e3
        ampRF = 1.00
        ampIF = 1.00
        delay_ph=-0.*_np.pi #time delay in phase shift
        delay_t=delay_ph/(2.0*_np.pi*(df)) #time delay in seconds

        _np.random.seed()

        n_s=1201     
#        n_s=1200
        periods=20.0         
        tt=_np.linspace(0,periods/df,n_s)    
        tb = [tt[0], tt[-2]] #some way or another he does not like tt[-1] in the fftanal
#        n_s=4000001   
        fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))

        tmpRF = ampRF*_np.sin(2.0*_np.pi*(df)*tt)
#        tmpRF += 1.00*ampRF*_np.random.standard_normal( size=(tt.shape[0],) )   # there was an error here
        tmpRF += 0.05*ampRF*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )

        tmpIF = ampIF*_np.sin(2.0*_np.pi*(df)*(tt)+delay_ph)
#        tmpIF += 1.00*ampIF*_np.random.standard_normal( size=(tt.shape[0],) )
#        tmpIF += 1.00*ampIF*_np.random.uniform( low=-1, high=1, size=(tt.shape[0],) )
        tt_tb=[_np.where(tt<=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
        SignalTime=tt[tt_tb][1]-tt[tt_tb][0]
               
        #fft with sigle hanning window
        Xfft=_np.fft.fft(_np.hanning(len(tmpRF))*tmpRF,axis=0)
        Yfft=_np.fft.fft(_np.hanning(len(tmpRF))*tmpIF,axis=0)
        
        Gxxd=(2.0*Xfft*_np.conj(Xfft))*((1/fs)**2*(1/SignalTime))
        Gyyd=(2.0*Yfft*_np.conj(Yfft))*((1/fs)**2*(1/SignalTime))
        Gxyd=(2.0*Yfft*_np.conj(Xfft))*((1/fs)**2*(1/SignalTime))
        
        Gxx=Gxxd[0:int(_np.ceil(len(tt[tt_tb[0]:tt_tb[1]+1])/2.0))]
        Gyy=Gyyd[0:int(_np.ceil(len(tt[tt_tb[0]:tt_tb[1]+1])/2.0))]
        Gxy=Gxyd[0:int(_np.ceil(len(tt[tt_tb[0]:tt_tb[1]+1])/2.0))]
        
        fr=1/SignalTime
        fNQ=fs/2
        freqaxis=_np.arange(0,fNQ,fr)
        _plt.figure()
        _plt.plot(freqaxis,Gxy)
        
        cc=Gxy/_np.sqrt(Gxx*Gyy)
        ccbg=_np.mean(cc[-100:])
        sigmacc=_np.sqrt((1-_np.abs(cc)**2)**2/(2*1))
        _plt.figure()
    #    _plt.plot(freqaxis,cc)
        _plt.plot(freqaxis,cc-ccbg)
        _plt.plot(freqaxis,sigmacc,'--',color='k')
        
        MaxFreq=fs/2.
        print('Maximal frequency: '+str(MaxFreq)+' Hz')
        MinFreq=2.0*fs/n_s #2*fr
        print('Minimal frequency: '+str(MinFreq)+' Hz') 
        
        f1=0e3
        f2=20e3
        integrand=(_np.real(cc-ccbg)/(1-_np.real(cc-ccbg)))[_np.where(freqaxis>=f1)[0][0]:_np.where(freqaxis>=f2)[0][0]]
        integralfreqs=freqaxis[_np.where(freqaxis>=f1)[0][0]:_np.where(freqaxis>=f2)[0][0]]
        integral=_np.trapz(integrand,integralfreqs)
        Bvid=0.5e6
        Bif=200e6        
        sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
        sens = _np.sqrt(2*Bvid/Bif/sqrtNs) 
        Tfluct=_np.sqrt(2*integral/Bif)
        print('Tfluct/T= '+str(Tfluct))

    if not sintest:
        
        filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
        print(fils[ii])
        tt, tmpRF, tmpIF = \
            _np.loadtxt(filn, dtype=_np.float64, unpack=True, usecols=(0,1,2))
        tt = 1e-3*tt
        tt_tb=[_np.where(tt==tb[0])[0][0],_np.where(tt==tb[1])[0][0]]
        tt=tt[tt_tb[0]:tt_tb[1]+1]
        tmpRF=tmpRF[tt_tb[0]:tt_tb[1]+1]
        tmpIF=tmpIF[tt_tb[0]:tt_tb[1]+1]
    
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
    
        fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))
        SignalTime=tt[tt_tb][1]-tt[tt_tb][0]    
        
        #for multiple windows
        Navr=1000
        windowoverlap=0.5
        nsig=len(tmpRF)
        nwins=int(_np.floor(nsig*1.0/(Navr-Navr*windowoverlap + windowoverlap)))
        noverlap=int(nwins*windowoverlap)
        ist=_np.arange(Navr)*(nwins-noverlap)   #Not actually using 1000 windows due to side effects?
        
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
    
#        _plt.figure()
#        _plt.plot(freq,Pxy)
        
        cc=Pxy/_np.sqrt(Pxx*Pyy)
        ccbg=_np.mean(cc[-100:])
        sigmacc=_np.sqrt((1-_np.abs(cc)**2)**2/(2*Navr))
        
        MaxFreq=fs/2.0
        print('Maximal frequency: '+str(MaxFreq)+' Hz')
        MinFreq=2.0*fs/nwins #2*fr
        print('Minimal frequency: '+str(MinFreq)+' Hz') 
        
        _plt.figure()
    #    _plt.plot(freq,cc)
        _plt.plot(freq/1000,_np.real(cc-ccbg))
        _plt.plot(freq/1000,2*sigmacc,'--',color='red')
        _plt.ylabel(r'Re($\gamma$-$\gamma_{bg}$)')
        _plt.xlabel('frequency [kHz]')
        _plt.title('Real part of background subtracted complex coherence')
        _plt.axvline(MinFreq/1000, color='k')

        
        f1=MinFreq
        f2=100e3
        integrand=(_np.real(cc-ccbg)/(1-_np.real(cc-ccbg)))[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]
        integralfreqs=freq[_np.where(freq>=f1)[0][0]:_np.where(freq>=f2)[0][0]]
        integral=_np.trapz(integrand,integralfreqs)
        Bvid=0.5e6
        Bif=200e6        
        sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
        sens = _np.sqrt(2*Bvid/Bif/sqrtNs)  
        
        
        Tfluct=_np.sqrt(2*integral/Bif)
        sigmaTfluct=_np.sqrt(_np.sum((sigmacc*fr)**2))/(2*Bif*Tfluct)
        print('Tfluct/T= '+str(Tfluct)+'+- '+str(sigmaTfluct))
    #    
 
