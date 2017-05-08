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
from pybaseutils import fft_analysis as _fft
from pybaseutils import plt_utils as _pltut

# datafolder = _os.path.abspath(_os.path.join('..', 'bin'))
datafolder = _os.path.abspath( _os.path.join( '/', 'Volumes', 'NawName','HeliotronJ', 'DATA' ) )
# datafolder = _os.path.join('/homea','weir','bin')

print('Existence check: ' + datafolder)
if _os.path.exists(datafolder):
    print('Data folder exists')
else:
    print('Data folder does not exist')
# endif

cmPerGHz = 1
titls = list()

# =============== #
#Nw = 3000
#Nw = 1000
Nw = 15000
# Nw = 250

cohtitl = 'CECE_jan17'
titls.extend( [cohtitl + '_fix2'] )
titls.extend( [cohtitl + '_fix4'] )
clrs = ['b', 'g']
nb = [0.150, 0.170]
tb = [0.192, 0.370]
intb = [15e3, 400e3]  # original
#intb = [15e3, 500e3]  # original
#intb = [50e3, 400e3]  # broadband fluctuations
## intb = [400e3, 465e3]  # high frequency mode
#intb = [410e3, 450e3]  # high frequency mode

# ================ #

#cohtitl = 'CECE_jan27'
#titls.extend( [cohtitl + '_fix1'] )
#titls.extend( [cohtitl + '_fix2'] )
#titls.extend( [cohtitl + '_fix3'] )
#titls.extend( [cohtitl + '_fix4'] )
#clrs = ['b', 'g', 'm', 'r']
#nb = [0.150, 0.170]
#tb = [0.220, 0.310]
##intb = [213e3, 220e3]  # original
##intb = [60e3, 100e3]  # original
##intb = [410e3, 450e3]  # high frequency mode
#intb = [220e3, 450e3]  # noise

Tefluct = list()

# =============== #

Bvid = 0.5e6    # 
Bif = 0.180e9   # [GHz], channel bandwidth
sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
sens = _np.sqrt(2*Bvid/Bif/sqrtNs)
print('Sensitivity to Te fluctuations: %3.2f percent'%(100*sens,))

# =============== #

_hCxy = _plt.figure('Coherence Length')
_aCxy = _hCxy.gca()
_aCxy.set_xlabel('r [cm]')
_aCxy.set_ylabel(r'ln(C$_{xy}$)')
_aCxy.set_title('Coherence Length')

# =============== #

_mfig = _plt.figure('Mean Spectra')
msub1 = _plt.subplot(3, 1, 1)
msub1.set_xlabel('freq [KHz]')
msub1.set_ylabel(r'P$_{x,y}$')
msub1.set_title('Mean Spectra from Ensemble')    

msub2 = _plt.subplot(3, 1, 2, sharex=msub1)
# msub2.set_ylim((0, 1))
msub2.set_xlabel('freq [KHz]')
msub2.set_ylabel(r'C$_{x,y}$')

msub3 = _plt.subplot(3, 1, 3, sharex=msub1)
# msub3.set_ylim((0, 1))
msub3.set_xlabel('freq [KHz]')
msub3.set_ylabel(r'$\phi_{x,y}$ [rad]')
    
    
jj = 0
for scantitl in titls:
    print(scantitl)
    if scantitl == 'CECE_jan17_fix2':    
        fils = ['CECE.65625','CECE.65626','CECE.65627','CECE.65628','CECE.65629']
        freqs = [68.3,       68.0,        68.2,        68.1,         68.1]
        
        fils.extend(['CECE.65630','CECE.65631','CECE.65632','CECE.65633','CECE.65634'])
        freqs.extend([68.15,      68.05,       67.95,        67.90,      68.125])

#        xlims = (0,500)
#        ylims1 = (0,1.8)
        txtstr = ''
        
    elif scantitl == 'CECE_jan17_fix4':    
        fils = ['CECE.65642','CECE.65643','CECE.65644','CECE.65645','CECE.65646','CECE.65647']
        freqs = [68.0,       68.3,        68.2,         68.1,       68.15,       68.05]
        
        fils.extend(['CECE.65648','CECE.65649','CECE.65650','CECE.65651','CECE.65652'])
        freqs.extend([67.95,      68.25,       68.125,      67.90,       67.80])    

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
    
    nfils = len(fils)
    freqs = _np.asarray(freqs, dtype=_np.float64)
    
    Nxy = _np.zeros( (nfils,), dtype=_np.complex64)
    Pxy = _np.zeros( (nfils,), dtype=_np.complex64)
    Pxx = _np.zeros( (nfils,), dtype=_np.complex64)
    Pyy = _np.zeros( (nfils,), dtype=_np.complex64)
    Cxy = _np.zeros( (nfils,), dtype=_np.complex64)
    phxy= _np.zeros( (nfils,), dtype=_np.complex64)
    
    Te = _np.zeros( (nfils,), dtype=_np.float64)
    
    for ii in range(nfils):
        
        filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
        print(filn)
        tt, tmpRF, tmpIF = \
            _np.loadtxt(filn, dtype=_np.float64, usecols=(0,1,2), unpack=True)
            
        tt = 1e-3*tt
        j0 = int( _np.floor( (tb[0]-tt[0])/(tt[1]-tt[0])) )
        j1 = int( _np.floor( (tb[1]-tt[0])/(tt[1]-tt[0])) )
        if j0<=0:        j0 = 0        # end if
        if j1>=len(tt):  j1 = -1       # end if    

        n0 = int( _np.floor( (nb[0]-tt[0])/(tt[1]-tt[0])) )
        n1 = int( _np.floor( (nb[1]-tt[0])/(tt[1]-tt[0])) )        
        if n0<=0:        n0 = 0        # end if
        if n1>=len(tt):  n1 = -1       # end if    
        
#        NRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=nb, Navr=Nw*1/_np.floor((tb[1]-tb[0])/(nb[1]-nb[0])))      
        IRfft = _fft.fftanal(tt, tmpRF, tmpIF, tbounds=tb, Navr=Nw)      
        
        if ii == 0:
#            Nxy_avg = _np.zeros((len(NRfft.freq),), dtype=_np.complex64); Nxy_var = _np.zeros_like(Nxy_avg)            
            Pxy_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pxy_var = _np.zeros_like(Pxy_avg)
            Pxx_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pxx_var = _np.zeros_like(Pxx_avg)
            Pyy_avg = _np.zeros((len(IRfft.freq),), dtype=_np.complex64); Pyy_var = _np.zeros_like(Pyy_avg)            
        # endif
            
    #    ircor = xcorr(tmpRF[j0:j1], tmpIF[j0:j1])
    
        # ---------------- #
        
        freq = IRfft.freq
        i0 = int( _np.floor( (intb[0]-freq[0])/(freq[1]-freq[0])) )
        i1 = int( _np.floor( (intb[1]-freq[0])/(freq[1]-freq[0])) )   
        if i0<=0:          i0 = 0        # end if
        if i1>=len(freq):  i1 = -1       # end if
        
    #    sub1.plot(1e-3*IRfft.freq, 10*_np.log10(_np.abs(IRfft.Pxy)), '-')    
    #    sub1.set_ylabel('Cross Power [dB]')
    
        sub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(IRfft.Pxy), '-')        
#        sub1.plot(1e-3*NRfft.freq, 1e6*_np.abs(NRfft.Pxy), 'k-')                
        sub2.plot(1e-3*IRfft.freq, IRfft.Cxy, '-')
        sub3.plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')

#        Te[ii] = _np.sqrt(_np.trapz(_np.sqrt(IRfft.Cxy[i0:i1]), x=freq[i0:i1])/Bvid)        
        Te[ii] = _np.sqrt(_np.trapz(_np.sqrt(IRfft.Cxy[i0:i1]), x=freq[i0:i1])/Bif)
        print(100*Te)        

        # ---------------- #
        
        reP = _np.real(IRfft.Pxy)
        imP = _np.imag(IRfft.Pxy)
        
        Pxy[ii] = _np.trapz(reP[i0:i1], x=freq[i0:i1]) \
                    + 1j*_np.trapz(imP[i0:i1], x=freq[i0:i1])    
        Pxx[ii] = _np.trapz(IRfft.Pxx[i0:i1], x=freq[i0:i1])  # Pxx = trapz(freq, Pxx)
        Pyy[ii] = _np.trapz(IRfft.Pyy[i0:i1], x=freq[i0:i1])
        
        # ---------------- #

        # Running mean and variance (vs frequency)
        #    n = n + 1
        #    m_prev = m
        #    m = m + (x_i - m) / n
        #    S = S + (x_i - m) * (x_i - m_prev)
#        m_prev = Nxy_avg.copy()    
#        Nxy_avg += (NRfft.Pxy - Nxy_avg) / (ii+1)
#        Nxy_var += (NRfft.Pxy - Nxy_avg) * (NRfft.Pxy - m_prev)        
        
        m_prev = Pxy_avg.copy()    
        Pxy_avg += (IRfft.Pxy - Pxy_avg) / (ii+1)
        Pxy_var += (IRfft.Pxy - Pxy_avg) * (IRfft.Pxy - m_prev)        
        
        m_prev = Pxx_avg.copy()    
        Pxx_avg += (IRfft.Pxx - Pxx_avg) / (ii+1)
        Pxx_var += (IRfft.Pxx - Pxx_avg) * (IRfft.Pxx - m_prev)        
        
        m_prev = Pyy_avg.copy()    
        Pyy_avg += (IRfft.Pyy - Pyy_avg) / (ii+1)
        Pyy_var += (IRfft.Pyy - Pyy_avg) * (IRfft.Pyy - m_prev)                
    # end loop    
    Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) )
    Cxy = _np.sqrt( Cxy )
    phxy = _np.arctan2(_np.imag(Pxy), _np.real(Pxy))
    phxy[_np.where(phxy<2.7)] += _np.pi    
    phxy[_np.where(phxy>2.7)] -= _np.pi    

#    Cxy_avg = _np.abs( Pxy_avg.conjugate()*Pxy_avg ) / (_np.abs(Pxx_avg)*_np.abs(Pyy_avg) )
    [Cxy_avg, Cxy_var] = _fft.varcoh(Pxy_avg, Pxy_var, Pxx_avg, Pxx_var, Pyy_avg, Pyy_var) 
    Cxy_avg = _np.sqrt( Cxy_avg )
    Cxy_var = (0.5*Cxy_avg**(-0.5))**2.0*Cxy_var
         
    phxy_avg = _np.arctan2(_np.imag(Pxy_avg), _np.real(Pxy_avg))
    phxy_avg[_np.where(phxy_avg<2.7)] += _np.pi    
    phxy_avg[_np.where(phxy_avg>2.7)] -= _np.pi    
    phxy_var = _np.zeros_like(phxy_avg)
#    [phi_xy, fftinfo.varPhxy] = mean_angle(phixy_seg[0:Navr, :], varphi_seg[0:Navr,:], dim=0)  
        
    # msub1.errorbar(1e-3*IRfft.freq, 1e6*_np.abs(Pxy_avg), yerr=1e6*_np.sqrt(_np.abs(Pxy_var)), fmt='-')        
#    msub1.plot(1e-3*NRfft.freq, 1e6*_np.abs(Nxy_avg), '-', lw=2, color='k')        
    msub1.plot(1e-3*IRfft.freq, 1e6*_np.abs(Pxy_avg), '-', lw=2, color=clrs[jj])        
    msub1.fill_between(1e-3*IRfft.freq, 
                       1e6*(_np.abs(Pxy_avg)-_np.sqrt(_np.abs(Pxy_var))), 
                       1e6*(_np.abs(Pxy_avg)+_np.sqrt(_np.abs(Pxy_var))), 
                       facecolor=clrs[jj], alpha=0.15)
    ylims = msub1.get_ylim()
    msub1.set_ylim( 0, ylims[1] ) 
#    msub1.text(130, 1.6-jj*0.2, txtstr, color=clrs[jj])
#    msub1.set_xlim((0, 235.0))
#    msub1.set_ylim((0, 2.0))
    msub1.set_ylim( -0.05, ylims[1] ) 
    
    # msub2.errorbar(1e-3*IRfft.freq, Cxy_avg, yerr=_np.sqrt(Cxy_var), fmt='-')        
    msub2.plot(1e-3*IRfft.freq, _np.abs(Cxy_avg), '-', lw=2, color=clrs[jj])        
    msub2.fill_between(1e-3*IRfft.freq, 
                       _np.abs(Cxy_avg)-_np.sqrt(_np.abs(Cxy_var)), 
                       _np.abs(Cxy_avg)+_np.sqrt(_np.abs(Cxy_var)), 
                       facecolor=clrs[jj], alpha=0.15)
    ylims = msub2.get_ylim()
#    msub1.set_xlim((0, 235.0))
    msub2.set_ylim( -0.05, ylims[1] ) 
    msub2.text(130, 0.45-jj*0.075, txtstr, color=clrs[jj])
    
    msub3.plot(1e-3*IRfft.freq, phxy_avg, '-')
#    msub3.fill_between(1e-3*IRfft.freq, 
#                       1e6*(_np.abs(phxy_avg)-_np.sqrt(_np.abs(phxy_var))), 
#                       1e6*(_np.abs(phxy_avg)-_np.sqrt(_np.abs(phxy_var))), 
#                       facecolor=clrs[jj], alpha=0.5)
#    msub3.set_xlim((0, 235.0))
    
#    TeAvg = _np.sqrt(_np.trapz(Cxy_avg[i0:i1], x=freq[i0:i1])/Bif)    
    TeAvg, TeVar, _, _ = _ut.trapz_var(freq[i0:i1], Cxy_avg[i0:i1], varx=None, vary=_np.abs(Cxy_var[i0:i1]) )
#    TeAvg = _np.sqrt(TeAvg/Bvid)
#    TeVar = (0.5/Bif*(TeAvg/Bvid)**(-0.5))**2.0*TeVar
    TeAvg = _np.sqrt(TeAvg/Bif)
    TeVar = (0.5/Bif*(TeAvg/Bif)**(-0.5))**2.0*TeVar
    print(100*TeAvg, 100*_np.sqrt(TeVar))            
#
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

    scantitl += '_%ito%iKHz'%(int(1e-3*intb[0]), int(1e-3*intb[1]))
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
            verbose=True, dotsperinch = 300, transparency = True)    
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
            verbose=True, dotsperinch = 300, transparency = True)

    # ================== #
    scantitl += '_Cxy'

    # Te = _np.sqrt(_np.trapz(IRfft.Cxy[i0:i1], x=freq[i0:i1])/Bif)    
    Te = _np.sqrt(Cxy/Bif)    
    Tefluct.extend([_np.max(Te)])
    
    _plt.figure(cohphi_title)
    _plt.sca(_aCxyRF)
     
    _aCxyRF.plot(freqs, Cxy, 'o')
    ylims = _aCxyRF.get_ylim()
    _aCxyRF.set_ylim( 0, ylims[1] ) 
    _aCxyRF.text(_np.average(freqs), 1.15*_np.mean(Cxy), 
              '%i to %i KHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])), fontsize=12)
    _aCxyRF.text(_np.average(freqs), 0.95*_np.mean(Cxy), 
              'Te fluctuation level ~ %3.2f'%(100*Te.max(),), fontsize=12)
    _aCxyRFb.plot(freqs, phxy, 'rx')
     
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
            verbose=True, dotsperinch = 300, transparency = True)    
    _pltut.savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
            verbose=True, dotsperinch = 300, transparency = True)
    
    # ================ #

    _plt.figure('Coherence Length')
    _hCxy.sca(_aCxy)
    _aCxy.plot(_np.abs(freqs-68.0)*cmPerGHz, _np.log(Cxy), 'o', color=clrs[jj] )

    # ================ #
    
    jj += 1
# end shot lists

print('Done with loop over lists')

_plt.figure('Coherence Length')
_plt.sca(_aCxy)
#_aCxy.text(0.6*_np.abs(freqs-68.0)*cmPerGHz, 0.95*_np.mean(_np.log(Cxy)), 
#          '%i to %i KHz'%(int(1e-3*freq[i0]),int(1e-3*freq[i1])), fontsize=12)
#_aCxy.text(0.6*_np.abs(freqs-68.0)*cmPerGHz, 1.15*_np.mean(_np.log(Cxy)),
#          'Te fluctuation level ~ %3.2f'%(100*Tefluct,), fontsize=12)
    
cohtitl += '_%ito%iKHz'%(int(1e-3*intb[0]), int(1e-3*intb[1]))
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='png', close=False, 
        verbose=True, dotsperinch = 300, transparency = True)    
_pltut.savefig(_os.path.join(datafolder, cohtitl), ext='eps', close=False, 
        verbose=True, dotsperinch = 300, transparency = True)

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
    
print('All finished')


# =================================================================== #
# =================================================================== #


