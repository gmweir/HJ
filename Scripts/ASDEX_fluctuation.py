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
# =============================== #

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'Workshop'))
#datafolder = _os.path.abspath(_os.path.join('W://','HJ','Data'))
#datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

cmPerGHz = 1
Fs = 1e6
minFreq = 10e3
#minFreq = 15e3
#minFreq = 7.5e3
#minFreq = 0.1e3
fLPF=False
f0 = False
sepshots = False  # separate the shots into different figures

backgroundsubtract = True  # use the method with _np.real(cc-ccbg)
#backgroundsubtract = False  # automatically set ccbg = 0.0, no background subtraction

keepphase = True  # preserve the phase, convert to phasor notation then subtract amplitude of background
#keepphase = False # don't worry abotu the phase (assume small complex comp. of bg), just subtract off the whole background spectrum
#keepphase only matters if we use oldstylesubtract = False

oldstylesubtract = True  # reproduce results from ECE paper and Creely, etc. (background subtract coherence within frequency limits)
#oldstylesubtract = False  # more "Physics based" (background subtract noise spectra as measured pre-plasma start-up)

Bvid=0.5e6
Bif=200e6
#windowoverlap=0.5

# ============================================================== #
# ============================================================== #
fils = []
freqs = []
freq_ref = 60.0   # [GHz]
intb = [10e3, 100e3]  # good
#intb = [40e3, 110e3]  # good
bg = [400e3, 500e6]  # background for coherence subtraction - automatically selected with fallback


#freq_ref = 60.0   # [GHz]
#fils = ['CECE.69769','CECE.69770','CECE.69771','CECE.69772','CECE.69773','CECE.69777']
#freqs = [13.075,     13.075,      13.085,      13.095,      13.105,      13.08]
#freqs = [4.0*freq+8.0 for freq in freqs]
##tb=[0.15,0.25]   # background data
#tbg = 0.25  # background

#04-02-18 fix 1
#fils = ['CECE.69757','CECE.69758','CECE.69759','CECE.69760','CECE.69761','CECE.69762']
#freqs = [13.075,     13.095,      13.115,      13.055,      13.085,      13.045] 
#freqs = [4.0*freq+8.0 for freq in freqs]
#tb=[0.32,0.40]   # background data
#tbg = 0.27  # background

#07-02-18 fix 1
#fils = ['CECE.69884','CECE.69885','CECE.69886','CECE.69887','CECE.69888','CECE.69889','CECE.69890']
#freqs = [13.075,     13.062,      13.085,      13.050,      13.037,      13.025,       13.012] 
#freqs = [4.0*freq+8.0 for freq in freqs]
#tb=[0.32,0.40]   # background data
#tbg = 0.27  # background


#tb = [0.285, 0.315]
##intb = [10e3, 200e3]
#intb = [10e3, 275e3]  # good
##intb = [10e3, 400e3]
##bg = [200e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
#bg = [275e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
##bg = [400e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
#Bvid = intb[1]

#txtstr = 'jan17_fix4'
#fils = ['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647']
#freqs = [68.0,       68.3,        68.2,         68.1,       68.15,       68.05]
#
#fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
#freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])
#tbg = 0.17
##tb = [0.192, 0.370]  # a little bit of saturation at beginning
#tb = [0.20, 0.370]   # a little bit of saturation at beginning
##tb = [0.20, 0.250]   #
##tb = [0.25, 0.30]    # we need to check the log for this shot to figure out timing
##tb = [0.32, 0.37]    # we need to check the log for this shot to figure out timing
##tb=[0.3,0.39]
#
##intb = [15e3, 400e3]  # original
##bg = [400e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
#intb = [375e3, 500e3]  # much better!  with Gavin's method of background subtraction
#bg = [0.0e3, 375.0e3]  # background for coherence subtraction - automatically selected with fallback

## January 17th, 2017
#txtstr = 'jan17_fix1'
#freq_ref = 68.0   # [GHz]
#fils = ['CECE.65624','CECE.65625']
#freqs = [68.3,       68.3]
#tbg = 0.17  # background
#tb = [0.20, 0.37] #time bounds
##intb = [10e3, 275e3]  # good
##bg = [275e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
#intb = [375e3, 500e3]  # good
#bg = [10e3, 375e6]  # background for coherence subtraction - automatically selected with fallback
##Bvid = intb[1]

#txtstr = 'jan17_fix2'
freq_ref = 68.0   # [GHz]
fils = ['CECE.65626','CECE.65627','CECE.65628','CECE.65629']
freqs = [68.0,        68.2,        68.1,         68.1]

fils.extend(['CECE.65630','CECE.65631','CECE.65632','CECE.65633','CECE.65634'])
freqs.extend([68.15,      68.05,       67.95,        67.90,      68.125])
tb=[0.37,0.395]   # time bounds
tbg = 0.17  # background


#txtstr = 'jan17_fix3'
#freq_ref = 68.0   # [GHz]
#fils = ['CECE.65638','CECE.65639','CECE.65640','CECE.65641']
#freqs = [68.3,       68.3,        68.3,         68.3]


#txtstr = 'Jan 27, ECH: 107 kW'
#fils = ['CECE.65947','CECE.65948','CECE.65949','CECE.65950']
#freqs = [68.3,        68.3,         68.3,       68.3]
#tbg = 0.180  # background
#tb = [0.21, 0.315]
#
#txtstr = 'Jan. 27, ECH: 174 kW'
#fils = ['CECE.65953','CECE.65954','CECE.65955','CECE.65956','CECE.65957','CECE.65958']
#freqs = [68.3,       68.3,        68.3,         68.3,       68.3,        68.3]
#tbg = 0.170  # background
#tb = [0.23, 0.30]

#
#txtstr = 'Jan. 27, ECH: 236 kW'
#fils = ['CECE.65961','CECE.65962','CECE.65963','CECE.65964','CECE.65965']
#freqs = [68.3,       68.3,        68.3,         68.3,       68.3]
#tbg = 0.170  # background
#tb = [0.25, 0.34]
#
#txtstr = 'Jan. 27 ECH: 301 kW'
#fils = ['CECE.65968','CECE.65969','CECE.65971','CECE.65973']
#freqs = [68.3,       68.3,        68.3,         68.3]
#tbg = 0.170  # background
#tb = [0.25, 0.34]

#txtstr = 'Date unknown' #very quiescent data
#fils = ['CECE.99103','CECE.99104','CECE.99105','CECE.99106','CECE.99107',
#        'CECE.99108','CECE.99109','CECE.99110','CECE.99111','CECE.99112',
#        'CECE.99113','CECE.99114']
#freqs = [68.3, 68.2, 68.2, 68.1, 68.0, 68.15, 68,05, 67.95, 67.90, 68.125, 68.25, 68.80]
#tbg = 0.20  # background
#tb = [0.20, 0.40]

## october, 2015 - fix 1
#fils = ['CECE.60342','CECE.60343','CECE.60344']
#freqs = [67.400,       67.400,       67.400]
#tbg = 0.175
#tb = [0.24, 0.315]
#
### october, 2015 - fix 2
##fils = ['CECE.60349','CECE.60350','CECE.60351']
##freqs = [67.400,       67.400,       67.400]
##tbg = 0.175
##tb = [0.22, 0.350]
#
#intb = [15e3, 400e3]  # original
#bg = [400e3, 1e6]  # background for coherence subtraction - automatically selected with fallback
##intb = [375e3, 500e3]  # much better!  with Gavin's method of background subtraction
##bg = [0.0e3, 375.0e3]  # background for coherence subtraction - automatically selected with fallback

# ============================================================== #
# ============================================================== #


if f0:
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
    fLPF = max((min((2*intb[1], 400e3)), intb[1]))  # [Hz], frequency to reject
    Bvid = fLPF
#    lpf_order = 1       # Order of low pass filter
    lpf_order = 3       # Order of low pass filter
#    lpf_order = 6       # Order of low pass filter
    w0 = f0/(0.5*Fs)   # Normalized frequency

    # Design the LPF
    blpf, alpf = butter_lowpass(fLPF, 0.5*Fs, order=lpf_order)
#    blpf, alpf = _sig.butter(lpf_order, fLPF/(0.5*Fs), btype='low', analog=False)

    # Frequency response
    w, h = _sig.freqz(blpf,alpf)
    freq = w*Fs/(2.0*_np.pi)    # Frequency axis

nfils = len(fils)

freqs = _np.asarray(freqs[0:nfils], dtype=_np.float64)
Tefluct = []
sigmaTe = []
CC2 = []
_Pxx, _Pyy, _Pxy = 0.0, 0.0, 0.0


ylims = [0,0]
for ii in range(nfils):
    filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
    print(filn)
    _, shotno = filn.split('.')

    tt, tmpRF, tmpIF = \
        _np.loadtxt(filn, dtype=_np.float64, unpack=True, usecols=(0,1,2))
    tt = 1e-3*tt.copy()

    _plt.figure("raw data")
    _plt.plot(tt, tmpRF)
    _plt.plot(tt, tmpIF)


    if tbg>(tb[1]-tb[0]):
        dt = tb[1]-tb[0]
        # background signal part
        bgRF = tmpRF[_np.where((tt>tbg-dt)*(tt<tbg))].copy()
        bgIF = tmpIF[_np.where((tt>tbg-dt)*(tt<tbg))].copy()
        ttbg = tt[_np.where((tt>tbg-dt)*(tt<tbg))].copy()
    else:
        # background signal part
        bgRF = tmpRF[_np.where(tt<tbg)].copy()
        bgIF = tmpIF[_np.where(tt<tbg)].copy()
        ttbg = tt[_np.where(tt<tbg)].copy()
    # end if

    # signal part
    tt_tb=[_np.where(tt>=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
    tt=tt[tt_tb[0]:tt_tb[1]+1].copy()
    tmpRF=tmpRF[tt_tb[0]:tt_tb[1]+1].copy()
    tmpIF=tmpIF[tt_tb[0]:tt_tb[1]+1].copy()

    _plt.axvline(x=tt[0])
    _plt.axvline(x=tt[-1])

    tmpRF -= _np.mean(tmpRF)
    tmpIF -= _np.mean(tmpIF)
    bgRF -= _np.mean(bgRF)
    bgIF -= _np.mean(bgIF)
    if fLPF:
        tmpRF = _sig.filtfilt(blpf, alpf, tmpRF.copy())
        tmpIF = _sig.filtfilt(blpf, alpf, tmpIF.copy())
        bgRF = _sig.filtfilt(blpf, alpf, bgRF.copy())
        bgIF = _sig.filtfilt(blpf, alpf, bgIF.copy())

    if f0:
        # Apply a zero-phase digital filter to both signals
        tmpRF = _sig.filtfilt(b, a, tmpRF.copy())  # padding with zeros
        tmpIF = _sig.filtfilt(b, a, tmpIF.copy())  # padding with zeros
        bgRF = _sig.filtfilt(b, a, bgRF.copy())  # padding with zeros
        bgIF = _sig.filtfilt(b, a, bgIF.copy())  # padding with zeros

    if tt[1]-tt[0]!=tt[2]-tt[1]:
        tt2=_np.linspace(tt[0],tt[-1],len(tt),endpoint=True)
        tmpRF=_np.interp(_np.asarray(tt2,dtype=float), tt, tmpRF.copy())
        tmpIF=_np.interp(_np.asarray(tt2,dtype=float), tt, tmpIF.copy())
        tt=tt2.copy()

        tt2 = _np.linspace(ttbg[0], ttbg[-1], len(ttbg), endpoint=True)
        bgRF=_np.interp(_np.asarray(tt2,dtype=float), ttbg, bgRF.copy())
        bgIF=_np.interp(_np.asarray(tt2,dtype=float), ttbg, bgIF.copy())
        ttbg=tt2.copy()
    # end if

#        fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))
#        SignalTime=tt[tt_tb][1]-tt[tt_tb][0]  # you've already truncated the signal into time
    SignalTime=tt[-1]-tt[0]

    sig_anal = fftanal(tt.copy(), tmpRF.copy(), tmpIF.copy(), windowfunction='hanning',
                      onesided=True, minFreq=minFreq, plotit=False)
    sig_anal.fftpwelch()

    bg_anal = fftanal(ttbg.copy(), bgRF.copy(), bgIF.copy(), windowfunction='hanning',
                      onesided=True, minFreq=minFreq, plotit=False)
#                          onesided=True, Navr=sig_anal.Navr, plotit=False)
    bg_anal.fftpwelch()
    nwins = sig_anal.nwins
    fs = sig_anal.Fs
    fr = fs/float(nwins)
    Navr = sig_anal.Navr
    freq = sig_anal.freq.copy()
    Pxy = sig_anal.Pxy.copy()
    Pxx = sig_anal.Pxx.copy()
    Pyy = sig_anal.Pyy.copy()

    siglags = sig_anal.fftinfo.lags.copy()
    sigcorrcoef = sig_anal.fftinfo.corrcoef.copy()

    MaxFreq=fs/2.
    MinFreq=2.0*fs/nwins #2*fr

    print('Maximal frequency: '+str(MaxFreq)+' Hz')
    print('Minimal frequency: '+str(MinFreq)+' Hz')


    # integration frequencies
    f1=max((MinFreq, intb[0]))
    f2=min((MaxFreq, intb[1]))
    if1 = _np.where(freq>=f1)[0]
    if2 = _np.where(freq>=f2)[0]
    if1 = 0 if len(if1) == 0 else if1[0]
    if2 = len(freq) if len(if2) == 0 else if2[0]
    ifreqs = _np.asarray(range(if1, if2), dtype=int)

    # =================== #

    cc=Pxy/_np.sqrt(Pxx.real*Pyy.real)

    if backgroundsubtract:
        if oldstylesubtract:
            # background subtraction indices / frequencies
            ibg1 = _np.where(freq>=min((MaxFreq,bg[0])))[0]
            ibg2 = _np.where(freq>=bg[1])[0]
            ibg1 = 0 if len(ibg1) == 0 else ibg1[0]
            ibg2 = -1 if len(ibg2) == 0 else ibg2[0]

#            ibg1 = _np.where(10*_np.log10(_np.abs(Pxy))<-85)[0][0]
#            ibg2 = _np.where(10*_np.log10(_np.abs(Pxy))<-85)[0][-1]
            if freq[ibg1]<f2:
                df = int( (min((bg[1],freq[-1]))-bg[0])//(freq[1]-freq[0]))
                ibg1 = _np.where(freq>=min((MaxFreq,bg[0])))[0]
                ibg1 = ibg2-df if len(ibg1)==0 else ibg1[0]
            # end if
            backgroundfreqs=freq[ibg1:ibg2]

            # if we assume that Gnoise (Gxy shared noise can be represented as background as in Creely, et al)
##            ccbg=_np.mean(cc[-100:])
            ccbg=_np.mean(cc[ibg1:ibg2])
##            ccbg=cc[ibg1:ibg2].min()
##            ccbg = cc.min()
#            Pxy_tmp, Pxx_tmp, Pyy_tmp = (_np.trapz(bg_anal.Pxy[ibg1:ibg2]),
#                                         _np.trapz(bg_anal.Pxx[ibg1:ibg2]),
#                                         _np.trapz(bg_anal.Pyy[ibg1:ibg2]))
#            ccbg = Pxy_tmp /_np.sqrt(Pxx_tmp.real*Pyy_tmp.real)

        # =============================================== #
        else:
            # If we assume that the common noise between the channels, Gnoise
            # can be represented by the background spectra (instrument function),
            # possibly scaled to the signal
            bglags = bg_anal.fftinfo.lags.copy()
            bgcorrcoef = bg_anal.fftinfo.corrcoef.copy()
            if keepphase:
                fi = _np.arctan2(Pxy.real, Pxy.imag).copy()
                Amp = _np.abs(Pxy).copy()
                Amp_bg = _np.interp(freq, bg_anal.freq, _np.abs(bg_anal.Pxy))
                Pxy = (Amp-Amp_bg)*_np.exp(1j*fi)   # preserve the original phase
            else:
                # assume the magnitude of the complex part is small when it needs to be
                Pxy = Pxy.copy() - _np.interp(freq, bg_anal.freq, bg_anal.Pxy) # don't worry about maintaining phase, complex amplitude is small
            # end if
            Pxy_tmp = 0.0
            Pxx_tmp = 0.0
            Pyy_tmp = 0.0
#
#            # Scale the background common noise spectra to the magnitude of the signal
##            imax = -1
##            imax = _np.argmax(_np.abs(bg_anal.Pxy))
##            Ascale = _np.interp(bg_anal.freq[imax], freq, _np.abs(Amp))/_np.abs(bg_anal.Pxy[imax])
##            Pxy_tmp, Pxx_tmp, Pyy_tmp = Ascale*Pxy_tmp, Ascale*Pxx_tmp, Ascale*Pyy_tmp

            # Calculate the coherence after background subtraction
            # note that if Gnoise is Pxy_noise (background), and we keep Pxx_tmp=0, Pyy_tmp = 0
            # then this is exactly equation A10 from Creely, et. al. with a more
            # reasonable estimate for the common noise.
            cc = (Pxy-Pxy_tmp)/_np.sqrt((Pxx-Pxx_tmp).real*(Pyy-Pyy_tmp).real)
##            ccbg = _np.interp(freq, bg_anal.freq, ccbg)
            ccbg = 0.0*_np.ones_like(cc)
        # end if
        # =============================================== #
    else:
        ccbg = 0.0
    # end if
    Rxx = Pxx.copy()
    Ryy = Pyy.copy()
    Rxy = Pxy.copy()
    Rxx[1:-1, ...] *= 0.5
    Ryy[1:-1, ...] *= 0.5
    Rxy[1:-1, ...] *= 0.5
    if sig_anal.nwins%2:
        Rxx[-1, ...] *= 0.5
        Ryy[-1, ...] *= 0.5
        Rxy[-1, ...] *= 0.5
    Rxx = _np.fft.irfft(Rxx, n=sig_anal.nwins, axis=0)
    Ryy = _np.fft.irfft(Ryy, n=sig_anal.nwins, axis=0)
    Rxy = _np.fft.irfft(Rxy, n=sig_anal.nwins, axis=0)
    Rxx *= _np.sqrt(sig_anal.nwins)
    Ryy *= _np.sqrt(sig_anal.nwins)
    Rxy *= _np.sqrt(sig_anal.nwins)

    # Calculate the normalized auto- and cross-correlations
    Ex = Rxx[0, ...].copy()    # power in the x-spectrum, int( |u(f)|^2, df)
    Ey = Ryy[0, ...].copy()    # power in the y-spectrum, int( |v(f)|^2, df)

    Rxx = _np.fft.fftshift(Rxx, axes=0)
    Ryy = _np.fft.fftshift(Ryy, axes=0)
    Rxy = _np.fft.fftshift(Rxy, axes=0)

    Rxx /= Ex
    Ryy /= Ey
    corrcoef = Rxy/_np.sqrt(Ex*Ey)
    lags = (_np.asarray(range(1, sig_anal.nwins+1), dtype=int)-sig_anal.Nnyquist)/sig_anal.Fs

    _plt.figure("Correlation Coefficient")
    if backgroundsubtract and not oldstylesubtract: _plt.subplot(2,1,1)
    _plt.plot(1e6*siglags, sigcorrcoef, '-')
    _plt.ylabel(r'$\rho_{x,y}$')
    _plt.title('Correlation Coefficient')
    if backgroundsubtract and not oldstylesubtract:  
        _plt.plot(1e6*bglags, bgcorrcoef, 'k--')
        _plt.subplot(2,1,2)
        _plt.plot(1e6*lags, corrcoef, '-')
        _plt.xlabel("lags [us]")
        _plt.ylabel(r'$\rho_{x,y}$')
        _plt.title('background subtracted')

    # ======================== #

    sigmacc=_np.sqrt((1-_np.abs(cc)**2)**2/(2*Navr))
    rcc = _np.real(cc-ccbg)    # the real part is for creating an RMS
#        rcc = _np.abs(cc-ccbg)    # we already have an RMS, but whatever
    integrand=(rcc/(1.0-rcc))[ifreqs]
    integralfreqs=freq[ifreqs]
    integral=_np.trapz(integrand,integralfreqs)
#        integral *= Bvid/(integralfreqs[-1]-integralfreqs[0])
    sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
    sens = _np.sqrt(2*Bvid/Bif/sqrtNs)


    Tfluct=_np.sqrt(2*integral/Bif)


#        sigmaTfluct=_np.sqrt(_np.sum((sigmacc*fr)**2))/(2*Bif*Tfluct)

    sigmaTfluct=_np.sqrt(_np.sum((sigmacc*fr)**2))/(Bif*Tfluct)
#        print('Tfluct/T= '+str(100*Tfluct)+'%+- '+str(100*sigmaTfluct)+'%')
    msg = u'Tfluct/T=%2.3f\u00B1%2.3f%%'%(100*Tfluct, 100*sigmaTfluct)
    print(msg)
    Tefluct.append(Tfluct)
    sigmaTe.append(sigmaTfluct)

    # ============================================ #

    if sepshots:
        _plt.figure("Pxy.%s"%(shotno,))
    else:
        _plt.figure("Pxy")
    # end if
    _ax1 = _plt.subplot(3,1,1)
    _ax2 = _plt.subplot(3,1,2, sharex=_ax1)
    _ax3 = _plt.subplot(3,1,3, sharex=_ax1)
#        _ax1.plot(1e-3*freq,_np.abs(Pxy))
#        _ax1.ylabel(r'P$_{xy}$ [V$^2$/Hz]')
    _ax1.plot(1e-3*freq,10*_np.log10(_np.abs(Pxy)))
    if backgroundsubtract and not oldstylesubtract:  _ax1.plot(1e-3*bg_anal.freq,10*_np.log10(_np.abs(bg_anal.Pxy)), 'k--')
    _ax1.set_ylabel(r'P$_{xy}$ [dB/Hz]')
    _ax2.plot(1e-3*freq,10*_np.log10(_np.abs(Pxx)))
    if backgroundsubtract and not oldstylesubtract:  _ax2.plot(1e-3*bg_anal.freq,10*_np.log10(_np.abs(bg_anal.Pxx)), 'k--')
    _ax2.set_ylabel(r'P$_{xx}$ [dB/Hz]')
    _ax3.plot(1e-3*freq,10*_np.log10(_np.abs(Pyy)))
    if backgroundsubtract and not oldstylesubtract:  _ax3.plot(1e-3*bg_anal.freq,10*_np.log10(_np.abs(bg_anal.Pyy)), 'k--')
    _ax3.set_ylabel(r'P$_{yy}$ [dB/Hz]')
    _ax3.set_xlabel('f [KHz]')
    _ylims = _ax1.get_ylim()
    _ax1.axvline(x=1e-3*freq[ifreqs[0]], ymin=_ylims[0],
                 ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[0]]/_ylims[1],
                 linewidth=0.5, linestyle='--', color='black')
    _ax1.axvline(x=1e-3*freq[ifreqs[-1]], ymin=_ylims[0],
                 ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[-1]]/_ylims[1],
                 linewidth=0.5, linestyle='--', color='black')
#        if backgroundsubtract:
#            _ax1.axvline(x=1e-3*freq[ibg1], ymin=_ylims[0],
#                         ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[0]]/_ylims[1],
#                         linewidth=0.5, linestyle='--', color='red')
#            _ax1.axvline(x=1e-3*freq[ibg2], ymin=_ylims[0],
#                         ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[-1]]/_ylims[1],
#                         linewidth=0.5, linestyle='--', color='red')
#        # end if

    # ============================================ #

    if sepshots:
        _plt.figure("Cxy2.%s"%(shotno,))
    else:
        _plt.figure("Cxy2")
    # end if
    _plt.plot(1e-3*freq, _np.abs(cc))
#        _plt.plot(1e-3*bg_anal.freq,_np.abs(bg_anal.Cxy), 'k--')
    _plt.axhline(y=1.0/_np.sqrt(Navr), linestyle='--')
    _plt.xlabel('f [KHz]')
    _plt.ylabel(r'|$\gamma$|')
    _plt.title('RMS Coherence')

    if sepshots:
        _plt.figure("Cxy.%s"%(shotno,))
    else:
        _plt.figure("Cxy")
    # end if
#    _plt.plot(freq,cc)
    
    _plt.plot(1e-3*freq,rcc)
    _plt.plot(1e-3*freq,sigmacc,'--',color='red')
    if sepshots:
        ylims[1] = _plt.ylim()[1]
    else:
        ylims[1] = max((ylims[1], _plt.ylim()[1]))
    # end if

    _plt.ylim(tuple(ylims))
    _plt.axvline(x=1e-3*freq[ifreqs[0]], ymin=0, ymax=rcc[ifreqs[0]]/ylims[1],
                 linewidth=0.5, linestyle='--', color='black')
    _plt.axvline(x=1e-3*freq[ifreqs[-1]], ymin=0, ymax=rcc[ifreqs[-1]]/ylims[1],
                 linewidth=0.5, linestyle='--', color='black')
#        if backgroundsubtract:
#            _plt.axvline(x=1e-3*freq[ibg1], ymin=0, ymax=sigmacc[ifreqs[0]]/ylims[1],
#                         linewidth=0.5, linestyle='--', color='red')
#            _plt.axvline(x=1e-3*freq[ibg2], ymin=0, ymax=sigmacc[ifreqs[-1]]/ylims[1],
#                         linewidth=0.5, linestyle='--', color='red')
#        # end if
    _plt.xlabel('f [KHz]')
    _plt.ylabel(r'$\gamma_r-\gamma_{bg}$')

    msg = r'$\tilde{T}_e/<T_e>$=%2.3f'%(100*Tfluct,)+'\u00B1'+'%2.3f%%'%(100*sigmaTfluct,)
#        msg = r'$\tilde{T}_e/<T_e>$=%2.3f\u00B1%2.3f%%'%(100*Tfluct,100*sigmaTfluct,)
    _plt.text(x=1e-3*0.8*freq[ifreqs[-1]], y=0.8*rcc[ifreqs].max(), s=msg)

    # ============================================ #

#        if sepshots:
    if 1:
        cc2i = (_np.trapz(Pxy[ifreqs],integralfreqs)
            / _np.sqrt(_np.trapz(_np.abs(Pxx[ifreqs]),integralfreqs)*_np.trapz(_np.abs(Pyy[ifreqs]),integralfreqs)))

#            CC2.append(cc2i-ccbg)
        CC2.append(cc2i)
    # end if

    # ============================================ #

#        m_prev =  _np.copy(_Pxy)
    _Pxy += (Pxy.copy() - _Pxy) / (ii+1)
#        _Pxy_var += (Pxy.copy() - _Pxy) * (IRfft.Pxy - m_prev)

#        m_prev = _np.copy(_Pxx)
    _Pxx += (Pxx.copy() - _Pxx) / (ii+1)
#        Pxx_var += (Pxx.copy() - _Pxx) * (IRfft.Pxx - m_prev)

#        m_prev = _np.copy(_Pyy)
    _Pyy += (Pyy.copy() - _Pyy) / (ii+1)
#        Pyy_var += (IRfft.Pyy - Pyy_avg) * (IRfft.Pyy - m_prev)
# end if test
# end if

#if sepshots:
if 1:
    Tefluct = _np.asarray(Tefluct, dtype=_np.float64)
    CC2 = _np.asarray(CC2, dtype=_np.complex128)

    _plt.figure('RMS Coherence')
#    _plt.plot(freqs, _np.real(CC2), 'o' )
    _plt.plot(freqs, _np.abs(CC2), 'o' )
    _plt.axhline(y=1.0/_np.sqrt(Navr), linestyle='--')
    _plt.xlabel('freq [GHz]')
    _plt.ylabel(r'RMS Coherence')

    _plt.figure('Tfluct')
    _plt.errorbar(freqs, Tefluct, yerr=sigmaTe, fmt='o')
    _plt.axhline(sens, linestyle='--')
    _plt.xlabel('freq [GHz]')
    _plt.ylabel(r'$\tilde{T}_e/<T_e>$')
    _plt.draw()

# end if
# ========================================================================= #


_cc=_Pxy/_np.sqrt(_Pxx.real*_Pyy.real)
_ccbg = 0.0

# Treat all signals like they are one long plasma discharge
_sigmacc=_np.sqrt((1-_np.abs(_cc)**2)**2/(2*Navr))
_rcc = _np.real(_cc-_ccbg)    # the real part is for creating an RMS
#        rcc = _np.abs(cc-ccbg)    # we already have an RMS, but whatever
_integrand=(_rcc/(1.0-_rcc))[ifreqs]
_integral=_np.trapz(_integrand,integralfreqs)
#        integral *= Bvid/(integralfreqs[-1]-integralfreqs[0])
_sqrtNs = _np.sqrt(2*Bvid*nfils*(tb[-1]-tb[0]))
_sens = _np.sqrt(2*Bvid/Bif/_sqrtNs)

_Tfluct=_np.sqrt(2*_integral/Bif)
_sigmaTfluct=_np.sqrt(_np.sum((_sigmacc*fr/nfils)**2))/(Bif*_Tfluct)
#        print('Tfluct/T= '+str(100*Tfluct)+'%+- '+str(100*sigmaTfluct)+'%')
_msg = u'Tfluct/T=%2.3f\u00B1%2.3f%%'%(100*_Tfluct, 100*_sigmaTfluct)

_plt.figure("Pxy_all")
units = r"[I.U.$^2$/Hz]"
_ax1 = _plt.subplot(3,1,1)
_ax2 = _plt.subplot(3,1,2, sharex=_ax1)
_ax3 = _plt.subplot(3,1,3, sharex=_ax1)
#_ax1.plot(1e-3*freq,_np.abs(Pxy))
#_ax1.plot(1e-3*bg_anal.freq,_np.abs(bg_anal.Pxy), 'k--')
_ax1.semilogy(1e-3*freq,_np.abs(Pxy))
_ax1.semilogy(1e-3*bg_anal.freq,_np.abs(bg_anal.Pxy), 'k--')
_ax1.set_ylabel(r'P$_{xy}$ '+units)

#_ax2.plot(1e-3*freq, _np.abs(Pxx))
#_ax2.plot(1e-3*bg_anal.freq,_np.abs(bg_anal.Pxx), 'k--')
_ax2.semilogy(1e-3*freq, _np.abs(Pxx))
_ax2.semilogy(1e-3*bg_anal.freq,_np.abs(bg_anal.Pxx), 'k--')
_ax2.set_ylabel(r'P$_{xx}$ '+units)
#_ax3.plot(1e-3*freq, _np.abs(Pyy))
#_ax3.plot(1e-3*bg_anal.freq, _np.abs(bg_anal.Pyy), 'k--')
_ax3.semilogy(1e-3*freq, _np.abs(Pyy))
_ax3.semilogy(1e-3*bg_anal.freq, _np.abs(bg_anal.Pyy), 'k--')
_ax3.set_ylabel(r'P$_{yy}$ '+units)
_ax3.set_xlabel('f [KHz]')
_ylims = _ax1.get_ylim()
_ax1.axvline(x=1e-3*freq[ifreqs[0]], ymin=_ylims[0],
             ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[0]]/_ylims[1],
             linewidth=0.5, linestyle='--', color='black')
_ax1.axvline(x=1e-3*freq[ifreqs[-1]], ymin=_ylims[0],
             ymax=10*_np.log10(_np.abs(Pxy))[ifreqs[-1]]/_ylims[1],
             linewidth=0.5, linestyle='--', color='black')

