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

from pybaseutils import utils as _ut
#from pybaseutils import fft_analysis as _fft
from pybaseutils import plt_utils as _pltut
from FFT import fft_analysis as _fft
from scipy import signal as _sig
from FFT.fft_analysis import butter_lowpass
from FFT.notch_filter import iirnotch

# =============================================== #

# datafolder = _os.path.abspath(_os.path.join('..', 'bin'))
#datafolder = _os.path.abspath( _os.path.join( '/', 'Volumes', 'NawName','HeliotronJ', 'DATA' ) )
#datafolder = _os.path.abspath( _os.path.join('G:/', 'ownCloud','HeliotronJ','DATA' ) )
# datafolder = _os.path.join('/homea','weir','bin')
datafolder=_os.path.join('..','..','..')

print('Existence check: ' + datafolder)
if _os.path.exists(datafolder):
    print('Data folder exists')
else:
    print('Data folder does not exist')
# endif

# =============================================== #

cmPerGHz = 1
titls = list()

Fs = 1e6

# =============== #

#cohtitl = 'CECE_jan17'
##titls.extend( [cohtitl + '_fix2'] )
#titls.extend( [cohtitl + '_fix4'] )
cohtitl='local_test'
titls.extend([cohtitl+ 'Local Stef'])

#clrs = ['b', 'g']
#nb = [0.150, 0.174]
##tb = [0.192, 0.370]
#tb = [0.230, 0.370]
#
#f_hpf = 10e3
#Nw = _np.int((tb[1]-tb[0]) * _np.max((f_hpf,1e3)))
#Nwb = _np.int((nb[1]-nb[0]) * f_hpf)
##Nw = 3000
##Nw = 2000
##Nw = 1000
##Nw = 15000
##Nw = 250
#
#intb = [f_hpf, 400e3]
##intb = [15e3, 400e3]  # original
##intb = [15e3, 500e3]  # original
##intb = [50e3, 400e3]  # broadband fluctuations
### intb = [400e3, 465e3]  # high frequency mode
##intb = [410e3, 450e3]  # high frequency mode

# ================ #

#cohtitl = 'CECE_jan17'
#titls.extend( [cohtitl + '_fix1'] )
#titls.extend( [cohtitl + '_fix3'] )
#clrs = ['b', 'g', 'm', 'r', 'k']
#nb = [0.150, 0.175]
#tb = [0.200, 0.370]
#
#f_hpf = 10e3
##Nw = _np.int((tb[1]-tb[0]) * _np.max((f_hpf,10e3)))
#Nw = 1000
##Nwb = _np.int((nb[1]-nb[0]) * f_hpf)
##intb = [1e3, 20e3]  # low frequency fluctuations
##intb = [10e3, 400e3] # big peak
#intb = [25e3, 400e3] # big peak

# ================ #

#cohtitl = 'CECE_jan27'
#titls.extend( [cohtitl + '_fix1'] )
#titls.extend( [cohtitl + '_fix2'] )
#titls.extend( [cohtitl + '_fix3'] )
#titls.extend( [cohtitl + '_fix4'] )
#clrs = ['b', 'g', 'm', 'r']
#nb = [0.150, 0.170]
##tb = [0.260, 0.310]
##tb = [0.240, 0.340]   # some RF signal saturation after 310 on select shots
#tb = [0.220, 0.310]  # pretty safe bounds,  a little higher fluct. at start
#
#f_hpf = 360
#Nw = _np.int((tb[1]-tb[0]) * _np.max((f_hpf,10e3)))
#Nw = 750
#Nw = Nw; overlap = 0.50  # 0% overlap
##Nw = 2*Nw-1; overlap = 0.50  # 50% overlap
###Nw = 3*Nw; overlap = 0.75    # 75% overlap
##Nwb = _np.int((nb[1]-nb[0]) * f_hpf)
## #intb = [1e3, 40e3]  # low frequency fluctuations
##intb = [1e3, 15e3]  # low frequency fluctuations
#intb = [40e3, 110e3] # big peak
##intb = [210e3, 220e3]  # original
###intb = [60e3, 100e3]  # original
###intb = [410e3, 450e3]  # high frequency mode
##intb = [220e3, 450e3]  # noise
#intbg = [300e3,500e3]

# =============== #
#
#cohtitl = 'CECE_oct302015'
#titls.extend( [cohtitl + '_fix1'] )
#titls.extend( [cohtitl + '_fix2'] )


clrs = ['b', 'g', 'm', 'r']
nb = [0.150, 0.180]
tb = [0.240, 0.315]

f_hpf = 360
Nw = _np.int((tb[1]-tb[0]) * _np.max((f_hpf,10e3)))
Nw = 8
#Nw = 750
Nw = Nw; overlap = 0.50  # 0% overlap
#Nw = 2*Nw-1; overlap = 0.50  # 50% overlap
#Nw = 3*Nw; overlap = 0.75    # 75% overlap
intb = [_np.max((f_hpf,1e3)), 300e3] # big peak
intbg = [400e3,500e3]

# =============== #
##
#cohtitl = 'CECE_feb072018'
##titls.extend( [cohtitl + '_fix1a'] )
##titls.extend( [cohtitl + '_fix1b'] )
##titls.extend( [cohtitl + '_fix2a'] )   # including freq scans
##titls.extend( [cohtitl + '_fix2b'] )   # including freq scans
#titls.extend( [cohtitl + '_fixed2a'] )   # no freq scans
#titls.extend( [cohtitl + '_fixed2b'] )   # no freq scans
#clrs = ['b', 'g', 'm', 'r']
#nb = [0.150, 0.170]
##tb = [0.250, 0.350]
##tb = [0.220, 0.310]
##tb = [0.290, 0.315]
#tb = [0.285, 0.315]
#
#f_hpf = 360
#Nw = _np.int((tb[1]-tb[0]) * _np.max((f_hpf,10e3)))
#Nw = 750
#Nw = Nw; overlap = 0.50  # 0% overlap
##Nw = 2*Nw-1; overlap = 0.50  # 50% overlap
##Nw = 3*Nw; overlap = 0.75    # 75% overlap
##Nwb = _np.int((nb[1]-nb[0]) * f_hpf)
##intb = [1e3, 20e3]  # low frequency fluctuations
##intb = [_np.max((f_hpf,10e3)), 250e3] # big peak
##intbg = [250e3,500e3]
#intb = [_np.max((f_hpf,1e3)), 300e3] # big peak
##intbg = [300e3,500e3]
#intbg = [400e3,500e3]
#
# =============== #

fluctbw = 0.3e6
Bvid = 0.5e6    #
Bif = 0.200e9   # [GHz], channel bandwidth

Bvid = min(fluctbw, Bvid)  # minimum of fluctuation bandwidth and hardware

# =============== #
# =============== #

ylimsC = [-0.05, 1e-20]
ylimsP = [0.0, 1e-20]
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

fLPF=True
if fLPF:
    fLPF = 0.5*intb[1]      # [Hz], frequency to reject #was 2 but this did not work
    Bvid = fLPF
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

# =============== #

sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
sens = _np.sqrt(2*Bvid/Bif/sqrtNs)
print('Sensitivity to Te fluctuations: %3.2f percent'%(100*sens,))
# =============== #

sqrtNs = list()
sens = list()
Tefluct = list()

# =============== #

#_hCxy = _plt.figure('Coherence Length')
#_aCxy = _hCxy.gca()
#_aCxy.set_xlabel('r [cm]')
#_aCxy.set_ylabel(r'ln(C$_{xy}$)')
#_aCxy.set_title('Coherence Length')

# =============== #

_mfig = _plt.figure('Mean Spectra')
msub1 = _plt.subplot(3, 1, 1)
msub1.set_xlabel('freq [KHz]')
msub1.set_ylabel(r'P$_{x,y}$')
#msub1.set_ylabel('Cross Power [dB]')
msub1.set_title('Mean Spectra from Ensemble')

msub2 = _plt.subplot(3, 1, 2, sharex=msub1)
# msub2.set_ylim((0, 1))
msub2.set_xlabel('freq [KHz]')
#msub2.set_ylabel(r'C$_{x,y}$')
msub2.set_ylabel(r'$\gamma-\gamma_{bg}$')

msub3 = _plt.subplot(3, 1, 3, sharex=msub1)
# msub3.set_ylim((0, 1))
msub3.set_xlabel('freq [KHz]')
msub3.set_ylabel(r'$\phi_{x,y}$ [rad]')

_hCorr = _plt.figure('Cross Correlation')
_aCorr = _hCorr.gca()
_aCorr.set_xlabel('tlag [us]')
_aCorr.set_ylabel(r'$\rho_{x,y}$')
_aCorr.set_title('Cross-Correlation')

_h1Corr = _plt.figure('Cross Correlation 2')
_a1Corr = _h1Corr.gca()
_a1Corr.set_xlabel('tlag [ms]')
_a1Corr.set_ylabel(r'$\rho_{x,y}$')
_a1Corr.set_title('Cross-Correlation: 68.0 and 68.3 GHz')

#    corr_title = scantitl+' Correlation'
#    _hCorrRF = _plt.figure(cohphi_title)
#    _aCorrRF = _hCxyRF.gca()
#    # _aCxyRF.set_ylim((0, 1))
#    _aCorrRF.set_xlabel('freq [GHz]')
#    _aCorrRF.set_ylabel(r'xcorr')
#    _aCorrRF.set_title(scantitl)

_hdat = _plt.figure('RawData')
_a211 = _plt.subplot(2,1,1)
#_a211.set_xlabel('t [ms]')
_a211.set_ylabel('RF [V]')
_a211.set_title('CECE Signals')
_a212 = _plt.subplot(2,1,2, sharex=_a211, sharey=_a211)
_a212.set_xlabel('t [ms]')
_a212.set_ylabel('IF [V]')

jj = 0
Te = list()
Telim = list()
TeAvg = list()
statlim = list()
for scantitl in titls:
    print(scantitl)
    if scantitl == 'Local Stef':
        freq_ref = 60.0   # [GHz]
        fils = ['CECE.69769','CECE.69770','CECE.69771','CECE.69772','CECE.69773','CECE.69777']
        freqs = [13.075,     13.075,      13.085,      13.095,      13.105,      13.08]
        freqs = [4.0*freqq+8.0 for freqq in freqs]
        
        txtstr = ''
    
    if scantitl == 'CECE_jan17_fix1':     # almost all w/in BW of channels
        fils = ['CECE.65624','CECE.65625']
        freqs = [68.3,       68.3]

        txtstr = ''

    elif scantitl == 'CECE_jan17_fix2':         # almost all w/in BW of channels
        fils = ['CECE.65626','CECE.65627','CECE.65628','CECE.65629']
        freqs = [68.0,        68.2,        68.1,         68.1]

        fils.extend(['CECE.65630','CECE.65631','CECE.65632','CECE.65633','CECE.65634'])
        freqs.extend([68.15,      68.05,       67.95,        67.90,      68.125])

#        xlims = (0,500)
#        ylims1 = (0,1.8)
        txtstr = ''

    elif scantitl == 'CECE_jan17_fix3':     # almost all w/in BW of channels
        fils = ['CECE.65638','CECE.65639','CECE.65640','CECE.65641']
        freqs = [68.3,       68.3,        68.3,         68.3]

        txtstr = ''

    elif scantitl == 'CECE_jan17_fix4':     # almost all w/in BW of channels
        fils = ['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647']
        freqs = [68.0,       68.3,        68.2,         68.1,       68.15,       68.05]

#        fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
#        freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])

#        xlims = (0,500)
#        ylims1 = (0,1.8)
        txtstr = ''

    elif scantitl == 'CECE_jan27_fix1':
        fils = ['CECE.65947','CECE.65948','CECE.65949','CECE.65950']
        freqs = [68.3,        68.3,         68.3,       68.3]
#        xlims = (40,230)
#        ylims1 = (0,1.3)
        txtstr = 'ECH: 107 kW'
    elif scantitl == 'CECE_jan27_fix2':
        fils = ['CECE.65953','CECE.65954','CECE.65955','CECE.65956','CECE.65957','CECE.65958']
        freqs = [68.3,       68.3,        68.3,         68.3,       68.3,        68.3]
#        xlims = (40,230)
#        ylims1 = (0,1.3)
        txtstr = 'ECH: 174 kW'
    elif scantitl == 'CECE_jan27_fix3':
        fils = ['CECE.65961','CECE.65962','CECE.65963','CECE.65964','CECE.65965']
        freqs = [68.3,       68.3,        68.3,         68.3,       68.3]
#        xlims = (40,230)
#        ylims1 = (0,1.3)
        txtstr = 'ECH: 236 kW'
    elif scantitl == 'CECE_jan27_fix4':
        fils = ['CECE.65968','CECE.65969','CECE.65971','CECE.65973']
        freqs = [68.3,       68.3,        68.3,         68.3]
#        xlims = (40,230)
#        ylims1 = (0,1.3)
        txtstr = 'ECH: 301 kW'

    elif scantitl == 'CECE_feb072018_fix1a':
        fils = []
        freqs = []
        txtstr = ''

    elif scantitl == 'CECE_feb072018_fix1b':
        fils = []
        freqs = []
        txtstr = ''

    elif scantitl == 'CECE_feb072018_fix2a':
        fils = ['CECE.69916', 'CECE.69917', 'CECE.69918', 'CECE.69919', 'CECE.69920']
        freqs = [     64.380,       64.300,       64.260,       64.220,       64.180]

        fils.extend(['CECE.69921', 'CECE.69922', 'CECE.69923'])
        freqs.extend([     64.140,       64.100,       64.050])
        txtstr = '5:3, 8 GHz IF'

    elif scantitl == 'CECE_feb072018_fixed2a':
        fils = ['CECE.69916', 'CECE.69917']
        freqs = [     64.380,       64.300]

        txtstr = '5:3, 8 GHz IF'

    elif scantitl == 'CECE_feb072018_fix2b':
        fils = ['CECE.69924', 'CECE.69925', 'CECE.69926', 'CECE.69927', 'CECE.69928']
        freqs = [     60.380,       60.300,       60.220,       60.140,       60.460]

        fils.extend(['CECE.69929', 'CECE.69930', 'CECE.69931'])
        freqs.extend([     60.060,       60.300,       60.300])
        txtstr = '5:3, 4 GHz IF'

    elif scantitl == 'CECE_feb072018_fixed2b':
        fils = ['CECE.69925', 'CECE.69930', 'CECE.69931']
        freqs = [60.300,       60.300,       60.300]
        txtstr = '5:3, 4 GHz IF'

    elif scantitl == 'CECE_oct302015_fix1':
        fils = ['CECE.60342','CECE.60343','CECE.60344']
        freqs = [67.400,       67.400,       67.400]
        txtstr = 'fix1'

    elif scantitl == 'CECE_oct302015_fix2':
        fils = ['CECE.60349','CECE.60350','CECE.60351']
        freqs = [67.400,       67.400,       67.400]
        txtstr = 'fix2'

    # endif
#    ylims2 = (0, 0.4)
#    ylims3 = (-_np.pi, _np.pi)

    spectra_title = scantitl+' Spectra'
    _hfig = _plt.figure(spectra_title)
    sub1 = _plt.subplot(3, 1, 1)
    sub1.set_xlabel('freq [KHz]')
    sub1.set_ylabel(r'P$_{x,y}$')
    sub1.set_title(scantitl)
#    sub1.set_xlim(xlims)
#    sub1.set_ylim(ylims1)

    sub2 = _plt.subplot(3, 1, 2, sharex=sub1)
    # sub2.set_ylim((0, 1))
    sub2.set_xlabel('freq [KHz]')
    sub2.set_ylabel(r'C$_{x,y}$')
#    sub1.set_xlim(xlims)
#    sub1.set_ylim(ylims2)

    sub3 = _plt.subplot(3, 1, 3, sharex=sub1)
    # sub3.set_ylim((0, 1))
    sub3.set_xlabel('freq [KHz]')
    sub3.set_ylabel(r'$\phi_{x,y}$ [rad]')
#    sub1.set_xlim(xlims)
#    sub1.set_ylim(ylims3)

#    sub4 = _plt.subplot(4, 1, 4, sharex=sub1)
#    # sub3.set_ylim((0, 1))
#    sub4.set_xlabel('lags [ms]')
#    sub4.set_ylabel(r'xcorr')

    # =========== #

    cohphi_title = scantitl+' Coherence and Phase'
    _hCxyRF = _plt.figure(cohphi_title)
    _aCxyRF = _hCxyRF.gca()
    # _aCxyRF.set_ylim((0, 1))
    _aCxyRF.set_xlabel('freq [GHz]')
    _aCxyRF.set_ylabel(r'C$_{x,y}$')
    _aCxyRF.set_title(scantitl)
    _aCxyRFb = _aCxyRF.twinx()
    _aCxyRFb.set_ylabel(r'$\phi_{x,y}$ [rad]', color='r')

    corr_title = scantitl+' Correlation'
    _hCorrRF = _plt.figure(corr_title)
    _aCorrRF = _hCorrRF.gca()
    # _aCxyRF.set_ylim((0, 1))
    _aCorrRF.set_xlabel('lags [us]')
    _aCorrRF.set_ylabel(r'$\rho_{x,y}$')
    _aCorrRF.set_title(scantitl)

    nfils = len(fils)
    freqs = _np.asarray(freqs, dtype=_np.float64)

#    Nxy = _np.zeros( (nfils,), dtype=_np.complex128)
    Pxy = _np.zeros( (nfils,), dtype=_np.complex128)
    Cxy = _np.zeros( (nfils,), dtype=_np.complex128)
#    Corr= _np.zeros( (nfils,), dtype=_np.complex128)
    Pxx = _np.zeros( (nfils,), dtype=_np.float64)
    Pyy = _np.zeros( (nfils,), dtype=_np.float64)
    phxy= _np.zeros( (nfils,), dtype=_np.float64)

#    Te = _np.zeros( (nfils,), dtype=_np.float64)
#    Telim = _np.copy(Te)

    for ii in range(nfils):

        filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
        print(filn)
        if fils[ii].find('CECE.65642')>-1 or fils[ii].find('CECE.65643')>-1:
            plotalone=True
        else:
            plotalone=False
        # endif

        tt, tmpRF, tmpIF = \
            _np.loadtxt(filn, dtype=_np.float64, usecols=(0,1,2), unpack=True)

        tt = 1e-3*tt

        # =============== #
        if ii == 0:
            sqrtNs.extend([_np.sqrt(2*Bvid*(tb[-1]-tb[0]))])
            #_np.sqrtNs = _np.sqrt(2*intb[1]*(tb[-1]-tb[0]))
            sens.extend([_np.sqrt(2*Bvid/Bif/sqrtNs[-1])])
            print('Sensitivity to Te fluctuations: %3.2f percent'%(100*sens[-1],))
            print('Number of unique samples: %i'%(_np.floor(sqrtNs[-1]**2.0),))
        # end if

        j0 = int( _np.floor( (tb[0]-tt[0])/(tt[1]-tt[0])) )
        j1 = int( _np.floor( (tb[1]-tt[0])/(tt[1]-tt[0])) )
        if j0<=0:        j0 = 0        # end if
        if j1>=len(tt):  j1 = -1       # end if

        n0 = int( _np.floor( (nb[0]-tt[0])/(tt[1]-tt[0])) )
        n1 = int( _np.floor( (nb[1]-tt[0])/(tt[1]-tt[0])) )
        if n0<=0:        n0 = 0        # end if
        if n1>=len(tt):  n1 = -1       # end if

        # =================== #

#        _a211.plot(1e3*tt, tmpRF, '-', color=clrs[ii])
#        _a212.plot(1e3*tt, tmpIF, '-', color=clrs[ii])
        _a211.plot(1e3*tt, tmpRF, '-')
        _a212.plot(1e3*tt, tmpIF, '-')
#        _a211.set_xlim((270, 320))
        # =================== #

        if f0:
            # Apply a zero-phase digital filter to both signals
            tmpRF = _sig.filtfilt(b, a, tmpRF)  # padding with zeros
            tmpIF = _sig.filtfilt(b, a, tmpIF)  # padding with zeros
#            tmpRF = _sig.filtfilt(b, a, tmpRF, method='gust')  # Gustafson's method, no padding
#            tmpIF = _sig.filtfilt(b, a, tmpIF, method='gust')  # Gustafson's method, no padding

        if fLPF:
            tmpRF = _sig.filtfilt(blpf, alpf, tmpRF)
            tmpIF = _sig.filtfilt(blpf, alpf, tmpIF)
#            tmpRF = _sig.filtfilt(blpf, alpf, tmpRF, method='gust')
#            tmpIF = _sig.filtfilt(blpf, alpf, tmpIF, method='gust')

#        NRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=nb, Navr=Nwb)
#        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=Nw)
#        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=Nw, windowoverlap=0.0)
#        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=2*Nw-1, windowoverlap=0.5)
#        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=3*Nw, windowoverlap=0.75)
#        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=Nw, windowoverlap=overlap)
        IRfft = _fft.fftanal(tt, tmpRF-tmpRF.mean(), tmpIF-tmpIF.mean(), tbounds=tb, Navr=Nw, windowoverlap=overlap)

        if ii == 0:
#            Nxy_avg = _np.zeros((len(NRfft.freq),), dtype=_np.complex64); Nxy_var = _np.zeros_like(Nxy_avg)
#            Pxy_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pxy_var = _np.zeros_like(Pxy_avg)
#            Pxx_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pxx_var = _np.zeros_like(Pxx_avg)
#            Pyy_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pyy_var = _np.zeros_like(Pyy_avg)

            Pxy_avg = 0.0
            Pxx_avg = 0.0
            Pyy_avg = 0.0
            Pxy_var = 0.0
            Pxx_var = 0.0
            Pyy_var = 0.0
        # endif

#        xCorr = _np.correlate(tmpRF[j0:j1]-tmpRF[j0:j1].mean(), tmpIF[j0:j1]-tmpIF[j0:j1].mean(), mode='full')
##        xCorr /= xCorr[len(tmpRF[j0:j1])-1]
#        xCorr /= (len(tmpRF[j0:j1]) * tmpRF[j0:j1].std() * tmpIF[j0:j1].std())
##        lags = _np.arange(-len(tmpRF[j0:j1])+1, len(tmpRF[j0:j1]))
#        lags = _np.asarray(range(len(tmpRF[j0:j1])), dtype=_np.float64)
#        lags -= 0.5*len(tmpRF[j0:j1])

        # ---------------- #

        freq = IRfft.freq
        i0 = int( _np.floor( (intb[0]-freq[0])/(freq[1]-freq[0])) )
        i1 = int( _np.round( (intb[1]-freq[0])/(freq[1]-freq[0])) )
        if i0<=0:          i0 = 0        # end if
        if i1>=len(freq):  i1 = -1       # end if

        bg0 = int( _np.floor( (intbg[0]-freq[0])/(freq[1]-freq[0])) )
        bg1 = int( _np.round( (intbg[1]-freq[0])/(freq[1]-freq[0])) )
        if bg0<=0:          bg0 = 0        # end if
        if bg1>=len(freq):  bg1 = -1       # end if

    #    sub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')
    #    sub1.set_ylabel('Cross Power [dB]')

        Cbg = _np.mean(IRfft.Cxy[bg0:bg1])
        sub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(IRfft.Pxy), '-')
#        sub1.plot(1e-3*NRfft.freq, 1e6*_np.abs(NRfft.Pxy), 'k-')
        sub2.plot(1e-3*IRfft.freq, _np.abs(IRfft.Cxy), '-')
#        sub2.plot(1e-3*IRfft.freq, _np.real(IRfft.Cxy-Cbg), '-')
        sub3.plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')

#        Cbg = _np.mean(_np.abs(IRfft.Cxy[bg0:bg1]))
#        Cbg = _np.mean(_np.real(IRfft.Cxy[bg0:bg1]))
        Te.extend( [_np.sqrt(_np.abs( (2.0/Bif)*_np.trapz(_np.real(IRfft.Cxy[i0:i1]-Cbg)/(1.0-_np.real(IRfft.Cxy[i0:i1]-Cbg)) , x=freq[i0:i1]) ))] )

#        Te[ii] = (2.0/Bif)*_np.trapz(_np.real(_np.abs(IRfft.Cxy[i0:i1])-_np.mean(_np.abs(IRfft.Cxy[bg0:bg1]))) \
#                    / (1.0 - _np.real(_np.abs(IRfft.Cxy[i0:i1])-_np.mean(_np.abs(IRfft.Cxy[bg0:bg1])))), x=freq[i0:i1])
#        Te[ii] = (2.0/Bif)*_np.trapz(_np.real(IRfft.Cxy[i0:i1]-_np.mean(IRfft.Cxy[bg0:bg1])) \
#                    / (1.0 - _np.real(IRfft.Cxy[i0:i1]-_np.mean(IRfft.Cxy[bg0:bg1]))), x=freq[i0:i1])
#        Te[ii] = _np.sqrt(Te[ii])
#        Te[ii] = _np.sqrt((1.0)/Bif * _np.trapz(_np.abs(IRfft.Cxy[i0:i1]), x=freq[i0:i1]) )
#        Te[ii] = _np.sqrt((intb[1]-intb[0])/Bif * _np.trapz(_np.abs(IRfft.Cxy[i0:i1]), x=freq[i0:i1]) )

#        Telim[ii] = 1.0/Bif *_np.trapz(\
#            _np.sqrt((1.0-IRfft.Cxy[i0:i1]**2.0)**2.0/IRfft.Navr), x=freq[i0:i1])
#        Telim[ii] = (intb[1]-intb[0])/Bif *_np.trapz(\
#            _np.sqrt((1.0-IRfft.Cxy[i0:i1]**2.0)**2.0/IRfft.Navr), x=freq[i0:i1])
#        Telim[ii] = _np.sqrt(Telim[ii])
#        Telim.extend( [ 1e-3*_np.sqrt( ((intb[1]-intb[0])/Bif) *_np.trapz(\
#            _np.sqrt(_np.abs((1.0-IRfft.Cxy[i0:i1]*_np.conj(IRfft.Cxy[i0:i1])))**2.0/Nw), x=freq[i0:i1]) ) ] )
#        Telim.extend( [ _np.sqrt( ((intb[1]-intb[0])/Bif) *_np.trapz(\
#            _np.sqrt((1.0-_np.real(IRfft.Cxy[i0:i1])**2.0)**2.0/IRfft.Navr), x=freq[i0:i1]) ) ] )
        Telim.extend( [ _np.sqrt( (1.0/Bif) *_np.trapz(\
            _np.sqrt((1.0-_np.real(IRfft.Cxy[i0:i1])**2.0)**2.0/IRfft.Navr), x=freq[i0:i1]) ) ] )

        print('Estimated fluctuation level: %3.2f percent'%(100*Te[-1],))
        print('Radiometer law limit: %3.2f percent'%(100*sens[-1],))
        print('Statistical limit: %3.2f percent'%(100*Telim[-1],))

        # ---------------- #

        npts = len(IRfft.freq)
        lags = _np.arange(-npts//2,npts//2)
#        if npts-2*(npts//2)==0: # even
#            lags = _np.arange(-npts//2,npts//2)
#        else: # odd
#            lags = _np.arange(-npts//2+1,npts//2)
#        # end if
        lags = -lags/float(Fs)
        xCorr = _np.sqrt(npts)*_np.fft.ifft(IRfft.Cxy, n=npts)   # TODO!:  take real part here after ifft
##        # Cross-correlation of one frequency component
##        xCorr = _np.sqrt(len(IRfft.freq[i0:i1]))*_np.fft.ifft(IRfft.Cxy[i0:i1], n=len(IRfft.freq[i0:i1]))
        xCorr = _np.fft.fftshift(xCorr)
#        lags, xCorr = _fft.ccf(tmpRF[j0:j1]-tmpRF[j0:j1].mean(), tmpIF[j0:j1]-tmpIF[j0:j1].mean(), Fs)
#        lags, xCorr = _fft.ccf_sh(tmpRF[j0:j1]-tmpRF[j0:j1].mean(), tmpIF[j0:j1]-tmpIF[j0:j1].mean(), Fs, Nw)
        if ii == 0:
#            Corr_avg = _np.zeros((len(lags),), dtype=_np.complex64); Corr_var = _np.zeros_like(Corr_avg)
            Corr_avg = 0.0
            Corr_var = 0.0
        # end if
        _aCorrRF.plot(1e6*lags, _np.abs(xCorr), '-')
#        _aCorrRF.plot(1e6*lags, xCorr, '.')

        # ---------------- #

#        reP = _np.real(IRfft.Pxy)
#        imP = _np.imag(IRfft.Pxy)
#
#        Pxy[ii] = _np.trapz(reP[i0:i1], x=freq[i0:i1]) \
#                    + 1j*_np.trapz(imP[i0:i1], x=freq[i0:i1])
        Pxy[ii] = _np.trapz(IRfft.Pxy[i0:i1], x=freq[i0:i1])  # Pxx = trapz(freq, Pxx)
        Pxx[ii] = _np.trapz(_np.real(IRfft.Pxx[i0:i1]), x=freq[i0:i1])  # Pxx = trapz(freq, Pxx)
        Pyy[ii] = _np.trapz(_np.real(IRfft.Pyy[i0:i1]), x=freq[i0:i1])

        # ---------------- #

        # Running mean and variance (vs frequency)
        #    n = n + 1
        #    m_prev = m
        #    m = m + (x_i - m) / n
        #    S = S + (x_i - m) * (x_i - m_prev)
#        m_prev = Nxy_avg.copy()
#        Nxy_avg += (NRfft.Pxy - Nxy_avg) / (ii+1)
#        Nxy_var += (NRfft.Pxy - Nxy_avg) * (NRfft.Pxy - m_prev)

        m_prev =  _np.copy(Pxy_avg)
        Pxy_avg += (IRfft.Pxy - Pxy_avg) / (ii+1)
        Pxy_var += (IRfft.Pxy - Pxy_avg) * (IRfft.Pxy - m_prev)

        m_prev = _np.copy(Pxx_avg)
        Pxx_avg += (IRfft.Pxx - Pxx_avg) / (ii+1)
        Pxx_var += (IRfft.Pxx - Pxx_avg) * (IRfft.Pxx - m_prev)

        m_prev = _np.copy(Pyy_avg)
        Pyy_avg += (IRfft.Pyy - Pyy_avg) / (ii+1)
        Pyy_var += (IRfft.Pyy - Pyy_avg) * (IRfft.Pyy - m_prev)

        m_prev = _np.copy(Corr_avg)
        Corr_avg += (xCorr - Corr_avg) / (ii+1)
        Corr_var += (xCorr - Corr_avg) * (xCorr - m_prev)
    # end loop
    ii = 0

#    Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) )
#    Cxy = _np.sqrt( Cxy )
    Cxy = Pxy / _np.sqrt(_np.abs(Pxx)*_np.abs(Pyy))  # integrated value

    phxy = _np.arctan2(_np.imag(Pxy), _np.real(Pxy))
    phxy[_np.where(phxy<2.7)] += _np.pi
    phxy[_np.where(phxy>2.7)] -= _np.pi

#    Cxy_avg = _np.abs( Pxy_avg.conjugate()*Pxy_avg ) / (_np.abs(Pxx_avg)*_np.abs(Pyy_avg) )
    [Cxy_avg, Cxy_var] = _fft.varcoh(Pxy=Pxy_avg, varPxy=Pxy_var, Pxx=Pxx_avg,
        varPxx=Pxx_var, Pyy=Pyy_avg, varPyy=Pyy_var, meansquared=False)
#    Cxy_avg = _np.sqrt( Cxy_avg )
#    Cxy_var = (0.5*Cxy_avg**(-0.5))**2.0*Cxy_var

    # Alternative Variance from A. White
#    Cxy_var = ((1.0-Cxy_avg**2.0)/_np.sqrt(2*IRfft.Navr))**2.0
    Cxy_var = ((1.0-Cxy_avg*_np.conj(Cxy_avg))/_np.sqrt(2*IRfft.Navr))**2.0

    phxy_avg = _np.arctan2(_np.imag(Pxy_avg), _np.real(Pxy_avg))
    phxy_avg[_np.where(phxy_avg<2.7)] += _np.pi
    phxy_avg[_np.where(phxy_avg>2.7)] -= _np.pi
    phxy_var = _np.zeros_like(phxy_avg)
#    [phi_xy, fftinfo.varPhxy] = mean_angle(phixy_seg[0:Navr, :], varphi_seg[0:Navr,:], dim=0)

    # Alternative:
#    phxy_var = (_np.sqrt(1.0-_np.abs(Cxy_avg)**2.0)/_np.sqrt(2.0*IRfft.Navr*_np.abs(Cxy_avg)))**2.0
    phxy_var = (_np.sqrt(1.0-_np.abs(Cxy_avg*_np.conj(Cxy_avg)))/_np.sqrt(2.0*IRfft.Navr*_np.abs(Cxy_avg)))**2.0

    # msub1.errorbar(1e-3*IRfft.freq, 1e6*_np.abs(Pxy_avg), yerr=1e6*_np.sqrt(_np.abs(Pxy_var)), fmt='-')
#    msub1.plot(1e-3*NRfft.freq, 1e6*_np.abs(Nxy_avg), '-', lw=2, color='k')
    msub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(Pxy_avg), '-', lw=2, color=clrs[jj])
    msub1.fill_between(1e-3*IRfft.freq,
                       1e6*(_np.abs(Pxy_avg)-_np.sqrt(_np.abs(Pxy_var))),
                       1e6*(_np.abs(Pxy_avg)+_np.sqrt(_np.abs(Pxy_var))),
                       facecolor=clrs[jj], alpha=0.15)
#    msub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(Pxy_avg)), '-', lw=2, color=clrs[jj])
#    msub1.fill_between(1e-3*IRfft.freq,
#                       10*_np.log10(_np.abs(Pxy_avg)-_np.sqrt(_np.abs(Pxy_var))),
#                       10*_np.log10(_np.abs(Pxy_avg)+_np.sqrt(_np.abs(Pxy_var))),
#                       facecolor=clrs[jj], alpha=0.15)
    ylims = msub1.get_ylim()
    ylimsP = [0.0, _np.max((ylims[1], ylimsP[1], 1.05*_np.max(1e6*(_np.abs(Pxy_avg)+_np.sqrt(_np.abs(Pxy_var))))))]
#    ylimsP = [0.0, _np.max((ylims[1], ylimsP[1], 1.05*_np.max(10*_np.log10(_np.abs(Pxy_avg)+_np.sqrt(_np.abs(Pxy_var))))))]
#    msub1.set_xlim((0, 235.0))
#    msub1.set_ylim( -0.05, ylims[1] )
    msub1.set_ylim( ylimsP[0], ylimsP[1] )
#    msub1.set_ylim( 0, ylims[1] )
#    msub1.text(130, 1.6-jj*0.2, txtstr, color=clrs[jj])
#    msub1.set_xlim((0, 235.0))
#    msub1.set_ylim((0, 2.0))
#    msub1.set_ylim( -0.05, ylims[1] )

    Cbg = _np.mean(Cxy_avg[bg0:bg1])
    # msub2.errorbar(1e-3*IRfft.freq, Cxy_avg, yerr=_np.sqrt(Cxy_var), fmt='-')
    msub2.plot(1e-3*IRfft.freq, _np.real(Cxy_avg-Cbg), '-', lw=2, color=clrs[jj])
    msub2.fill_between(1e-3*IRfft.freq,
                       _np.real(Cxy_avg-Cbg)-_np.sqrt(_np.abs(Cxy_var)),
                       _np.real(Cxy_avg-Cbg)+_np.sqrt(_np.abs(Cxy_var)),
                       facecolor=clrs[jj], alpha=0.15)
#    msub2.plot(1e-3*IRfft.freq, _np.real(Cxy_avg), '-', lw=2, color=clrs[jj])
#    msub2.fill_between(1e-3*IRfft.freq,
#                       _np.real(Cxy_avg)-_np.sqrt(_np.real(Cxy_var)),
#                       _np.real(Cxy_avg)+_np.sqrt(_np.real(Cxy_var)),
#                       facecolor=clrs[jj], alpha=0.15)
    ylims = msub2.get_ylim()
#    ylimsC = [-0.05, _np.max((ylims[1], ylimsC[1], 1.05*_np.max(_np.abs(Cxy_avg)+_np.sqrt(_np.abs(Cxy_var)))))]
    ylimsC = [-0.05, _np.max((ylims[1], ylimsC[1], 1.05*_np.max(_np.abs(Cxy_avg-Cbg)+_np.sqrt(_np.abs(Cxy_var)))))]
#    msub1.set_xlim((0, 235.0))
#    msub2.set_ylim( -0.05, ylims[1] )
    msub2.set_ylim( ylimsC[0], ylimsC[1] )
#    msub2.text(350, 0.75*ylimsC[1]-jj*ylimsC[1]/10, txtstr, color=clrs[jj])
#    msub2.text(130, 0.55-jj*0.075, txtstr, color=clrs[jj])
    msub2.text(130, 0.70-jj*0.075, txtstr, color=clrs[jj])
    msub2.axhline(y=1.0/_np.sqrt(IRfft.Navr), linestyle='--', linewidth=2, color='k')

    msub3.plot(1e-3*IRfft.freq, phxy_avg, '-')
    msub3.fill_between(1e-3*IRfft.freq,
                       (phxy_avg-_np.sqrt(phxy_var)),
                       (phxy_avg+_np.sqrt(phxy_var)),
                       facecolor=clrs[jj], alpha=0.5)
#    msub3.set_xlim((0, 235.0))

#    TeAvg.extend( [(2.0/Bif)*_np.trapz(_np.real(Cxy_avg[i0:i1]-_np.mean(Cxy_avg[bg0:bg1])) \
#                / (1.0 - _np.real(Cxy_avg[i0:i1]-_np.mean(Cxy_avg[bg0:bg1]))), x=freq[i0:i1])] )
#    TeAvg[jj] = _np.sqrt(TeAvg[jj])
#    TeAvg.extend([_np.sqrt(((intb[1]-intb[0])/Bif)*_np.trapz(_np.abs(Cxy_avg[i0:i1]), x=freq[i0:i1]))])
#    statlim.extend([_np.sqrt((intb[1]-intb[0])/Bif*_np.trapz((1.0-Cxy_avg[i0:i1]**2.0)**2.0/IRfft.Navr, x=freq[i0:i1]))])
#    TeAvg.extend([_np.sqrt(((1.0)/Bif)*_np.trapz(_np.abs(Cxy_avg[i0:i1]), x=freq[i0:i1]))])
#    statlim.extend([_np.sqrt((2.0)/Bif*_np.trapz((1.0-Cxy_avg[i0:i1]**2.0)**2.0/IRfft.Navr, x=freq[i0:i1]))])

    TeAvg = _np.abs(_np.trapz(_np.real(Cxy_avg[i0:i1]-Cbg)/(1.0-_np.real(Cxy_avg[i0:i1]-Cbg)) ,x=freq[i0:i1]))
    TeVar = _np.zeros_like(TeAvg)
    TeAvg = _np.sqrt(2.0*TeAvg/Bif)
    TeVar = (0.5*2.0/Bif*(2.0*TeAvg/Bif)**(-0.5))**2.0*TeVar
                ##    TeAvg = _np.sqrt(_np.trapz(Cxy_avg[i0:i1], x=freq[i0:i1])*(intb[1]-intb[0])/Bif)
#    TeAvg, TeVar, _, _ = _ut.trapz_var(freq[i0:i1], Cxy_avg[i0:i1], varx=None, vary=_np.abs(Cxy_var[i0:i1]) )
##    TeVar = (0.5*(intb[1]-intb[0])/Bif*(TeAvg)**(-0.5))**2.0*TeVar
#    print(100*TeAvg[jj])

#    NeAvg, NeVar, _, _ = _ut.trapz_var(freq[i0:i1], Cxy_avg[i0:i1], varx=None, vary=_np.abs(Cxy_var[i0:i1]) )
#    NeAvg = _np.sqrt(NeAvg/Bif)
#    NeVar = (0.5/Bif*(NeAvg/Bif)**(-0.5))**2.0*NeVar

    # --------------- #

    sub1.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
    sub1.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
    sub2.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
    sub2.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
    sub3.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
    sub3.axvline(x=1e-3*freq[i1], linewidth=2, color='k')

    # ================== #

    _plt.figure(spectra_title)
    _hfig.sca(sub1)
#    _hfig.tight_layout()

    scantitl += '_%ito%iKHz'%(int(1e-3*intb[0]), int(1e-3*intb[1]))
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='png', close=False,
            verbose=True, dotsperinch = 300, transparency = True)
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False,
            verbose=True, dotsperinch = 300, transparency = True)

    # ================== #
    scantitl += '_Cxy'

#    Tefluct.extend([_np.max(Te)])
#    Tefluct.extend([_np.mean(Te)])
    Tefluct.extend([TeAvg])

    _plt.figure(cohphi_title)
    _plt.sca(_aCxyRF)

    _aCxyRF.plot(freqs, Cxy, 'o')
    ylims = _aCxyRF.get_ylim()
    _aCxyRF.set_ylim( 0, ylims[1] )
    _aCxyRF.text(_np.average(freqs), 1.15*_np.mean(Cxy),
              '%i to %i KHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])), fontsize=12)
    _aCxyRF.text(_np.average(freqs), 0.95*_np.mean(Cxy),
              'Te fluctuation level ~ %3.2f'%(100*_np.max(Te),), fontsize=12)
    _aCxyRFb.plot(freqs, phxy, 'rx')

    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='png', close=False,
            verbose=True, dotsperinch = 300, transparency = True)
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False,
            verbose=True, dotsperinch = 300, transparency = True)

    # ================ #

#    _plt.figure('Coherence Length')
#    _hCxy.sca(_aCxy)
#    _aCxy.plot(_np.abs(freqs-68.0)*cmPerGHz, _np.log(Cxy), 'o', color=clrs[jj] )

    # ================ #

    # You can also calculate the average cross-correlation from the inverse FFT
    npts = len(IRfft.freq)
    lags = _np.arange(-npts//2,npts//2)
#    if npts-2*(npts//2)==0: # even
#        lags = _np.arange(-npts//2,npts//2)
#    else: # odd
#        lags = _np.arange(-npts//2+1,npts//2)
#    # end if
    lags = -lags/float(Fs)
    CorrAvg = _np.sqrt(npts)*_np.fft.ifft(Cxy_avg, n=npts)   # TODO!:  TAKE THE REAL PART AFTER INVERSE FFT
#        # Cross-correlation of one frequency component
#        xCorr = _np.sqrt(len(IRfft.freq[i0:i1]))*_np.fft.ifft(IRfft.Cxy[i0:i1], n=len(IRfft.freq[i0:i1]))
#    CorrAvg = _np.sqrt(len(IRfft.freq[i0:i1]))*_np.fft.ifft(Cxy_avg[i0:i1], n=len(IRfft.freq[i0:i1]))
#    lags = _np.arange(-len(IRfft.freq[i0:i1])+1, len(IRfft.freq[i0:i1]))
#    lags = -lags/float(Fs)
    CorrAvg = _np.fft.fftshift(CorrAvg)

    _plt.figure('Cross Correlation')
    _hCorr.sca(_aCorr)
    _aCorr.plot(1e6*lags, _np.abs(CorrAvg), '-', lw=2, color=clrs[jj] )
#    _aCorr.fill_between(1e6*lags,
#                       (Corr_avg-_np.sqrt(_np.abs(Corr_var))),
#                       (Corr_avg+_np.sqrt(_np.abs(Corr_var))),
#                       facecolor=clrs[jj], alpha=0.5)

    if plotalone:
        _plt.figure('Cross Correlation 2')
        _h1Corr.sca(_a1Corr)
        _a1Corr.plot(1e6*lags, _np.abs(CorrAvg), '-', lw=4, colors=clrs[jj])
    # end if
    # ================ #

    jj += 1
# end shot lists

print('Done with loop over lists')

_plt.figure('RawData')
_plt.sca(_a211)
_pltut.savefig(_os.path.join(datafolder, cohtitl+'_rawsig'), ext='png', close=False,
        verbose=True, dotsperinch = 300, transparency = True)
_pltut.savefig(_os.path.join(datafolder, cohtitl+'_rawsig'), ext='eps', close=False,
        verbose=True, dotsperinch = 300, transparency = True)


#_plt.figure('Coherence Length')
#_plt.sca(_aCxy)
##_aCxy.text(0.6*_np.abs(freqs-68.0)*cmPerGHz, 0.95*_np.mean(_np.log(Cxy)),
##          '%i to %i KHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])), fontsize=12)
##_aCxy.text(0.6*_np.abs(freqs-68.0)*cmPerGHz, 1.15*_np.mean(_np.log(Cxy)),
##          'Te fluctuation level ~ %3.2f'%(100*Tefluct,), fontsize=12)
#
#cohtitl += '_%ito%iKHz'%(int(1e-3*intb[0]), int(1e-3*intb[1]))
#_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='png', close=False,
#        verbose=True, dotsperinch = 300, transparency = True)
#_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='eps', close=False,
#        verbose=True, dotsperinch = 300, transparency = True)

# ========== #

_plt.figure('Mean Spectra')
msub1.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
msub1.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
msub2.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
msub2.axvline(x=1e-3*freq[i1], linewidth=2, color='k')
msub3.axvline(x=1e-3*freq[i0], linewidth=2, color='k')
msub3.axvline(x=1e-3*freq[i1], linewidth=2, color='k')

cohtitl += 'ensemble_spectra'
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='png', close=False,
        verbose=True, dotsperinch = 300, transparency = True)
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='eps', close=False,
        verbose=True, dotsperinch = 300, transparency = True)

# ========== #

_plt.figure('Cross Correlation')
_plt.sca(_aCorr)

cohtitl += '_xcorr'
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='png', close=False,
        verbose=True, dotsperinch = 300, transparency = True)
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='eps', close=False,
        verbose=True, dotsperinch = 300, transparency = True)

print('All finished')


# =================================================================== #
# =================================================================== #


