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
#from pybaseutils.plt_utils import savefig

# datafolder = _os.path.abspath(_os.path.join('~','bin'))
datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

cmPerGHz = 1

#scantitl = 'CECE_jan17_nelow'
# scantitl += '_50to400'
#scantitl += '_400to500'

ch = _np.asarray((_np.linspace(0,15,16)), dtype=_np.int64)
igch = [1,7,14,15,16]
igch = _np.asarray(igch, dtype=_np.int64)
msk = _np.ones(_np.shape(ch), dtype=_np.bool)
msk[igch-1] = False
ch = ch[msk]

LO = _np.asarray(56, dtype=_np.float64)     # [GHz]
IF = [2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5, 16.5, 
      3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]
IF = _np.asarray(IF, dtype=_np.float64)
ece_rf = LO+IF
ece_rf = ece_rf[msk]

ece_fils = ['ECEFAST.65624', 'ECEFAST.65625']
ech_fils = ['ECRH.65624', 'ECRH.65625']

#ece_fils = ['ECEFAST.65638', 'ECEFAST.65639', 'ECEFAST.65640', 'ECEFAST.65641']
#ech_fils = ['ECRH.65638', 'ECRH.65639', 'ECRH.65640', 'ECRH.65641']

fmod = 83
harms = [1, 2, 4, 5]
intb = [
        [65, 110],   # fundamental 
        [150, 180],  # second harmonic
        [320, 342],
        [399, 431]]
#        [570, 600]       
#        ]

tb = [0.190, 0.370]

hfig = _plt.figure()
sub1 = _plt.subplot(3, 1, 1)
sub1.set_xlabel('freq [KHz]')
sub1.set_ylabel('Cross Power')

sub2 = _plt.subplot(3, 1, 2, sharex=sub1)
# sub2.set_ylim((0, 1))
sub2.set_xlabel('freq [KHz]')
sub2.set_ylabel('Cross Coherence')

sub3 = _plt.subplot(3, 1, 3, sharex=sub1)
# sub3.set_ylim((0, 1))
sub3.set_xlabel('freq [KHz]')
sub3.set_ylabel('Cross Phase')    

# sub4 = _plt.subplot(4, 1, 4)
_hfig2 = _plt.figure('Cross-Phase / Coherence')
sub4 = _plt.gca()
# sub4.set_ylim((0, 1))
#sub4.set_xlabel('freq [GHz]')
sub4.set_xlabel('RF [GHz]')
sub4.set_ylabel('Phase(fmod)')    
sub4b = sub4.twinx()
sub4b.set_ylabel('Coherence(fmod)', color='r')

# sub4 = _plt.subplot(4, 1, 4)
_hfig3 = _plt.figure('ln[Amp] / Coherence')
sub5 = _plt.gca()
# sub5.set_ylim((0, 1))
#sub5.set_xlabel('freq [GHz]')
sub5.set_xlabel('RF [GHz]')
sub5.set_ylabel('ln[Amp](fmod)')    
sub5b = sub5.twinx()
sub5b.set_ylabel('Coherence(fmod)', color='r')

# --------------- #

nfils = len(ece_fils)
nch = len(ch) 
nharm = len(harms)
legs = []
# freqs = _np.asarray(freqs, dtype=_np.float64)

Pxy = _np.zeros( (nfils,nch,nharm), dtype=_np.complex64)
Pxx = _np.zeros( (nfils,nch,nharm), dtype=_np.complex64)
Pyy = _np.zeros( (nfils,nch,nharm), dtype=_np.complex64)
# Cxy = _np.zeros( (nfils,nch), dtype=_np.complex64)

for ii in range(nfils):
    
    filn = _os.path.abspath(_os.path.join(datafolder, ece_fils[ii]))
    print(filn)
    
    ece = _np.loadtxt(filn, dtype=_np.float64, unpack=False)
    tt = ece[:,0]
    ece = ece[:,1:]
    ece = ece[:,msk]

    # -------------- #
    # Scale to match Thomson here
    
    
    # -------------- #


    filn = _os.path.abspath(_os.path.join(datafolder, ech_fils[ii]))
    print(filn)
    
    # 
    ech = _np.loadtxt(filn, dtype=_np.float64, unpack=False)
    te = ech[:,0]
    Vbeam = ech[:,1]
    Vbody = ech[:,2]
    RF = ech[:,3]

    Vbeam = _np.interp(tt, te, Vbeam)
    Vbody = _np.interp(tt, te, Vbody)    
    RF = _np.interp(tt, te, RF)
        
    # 
    
    tt = 1e-3*tt
    j0 = int(_np.floor( 1 + (tb[0]-tt[0])/(tt[1]-tt[0])))
    j1 = int(_np.floor( 1 + (tb[1]-tt[0])/(tt[1]-tt[0])))        
    
    ref = RF[j0-5:j1+5]
    ref = ref/_np.max(ref)
    # ref -=_np.mean(ref[j0:j1])
    
    jj = -1
    for kk in ch:
        jj += 1
        
        sig = ece[j0-5:j1+5,jj]
        # sig -= _np.mean(sig[j0:j1])
        
        IRfft = fftanal(tt[j0-5:j1+5], ref, sig, tbounds=tb, Navr=4, overlap=0.85, 
                        nfft=5*(j1-j0), windowfunction='hanning')      

        # ---------------- #
           
        sub1.plot(1e-3*IRfft.freq, _np.abs(IRfft.Pxy), '-')    
        sub2.plot(1e-3*IRfft.freq, IRfft.Cxy, '-')
        sub3.plot(1e-3*IRfft.freq, IRfft.phi_xy, '-')
        
        # ---------------- #
    
        reP = _np.real(IRfft.Pxy)
        imP = _np.imag(IRfft.Pxy)

        freq = IRfft.freq
        for ff in range(nharm):

            ib0 = int(_np.round( (intb[ff][0]-freq[0])/(freq[1]-freq[0])))
            ib1 = int(_np.round( (intb[ff][1]-freq[0])/(freq[1]-freq[0])))   
            # ib1 += 1
            # ibf i1 == ib0 + 1: ib1 += 1 # endif


            Pxy[ii,jj,ff] = _np.trapz(reP[ib0:ib1+1], x=freq[ib0:ib1+1]) \
                            + 1j*_np.trapz(imP[ib0:ib1+1], x=freq[ib0:ib1+1])    
            Pxx[ii,jj,ff] = _np.trapz(IRfft.Pxx[ib0:ib1+1], x=freq[ib0:ib1+1])
            Pyy[ii,jj,ff] = _np.trapz(IRfft.Pyy[ib0:ib1+1], x=freq[ib0:ib1+1])

            if jj == 0:
                if ff == 0:
                    is0 = ib0
                    is1 = ib1
                # end if
                legs.extend( ['%3i Hz'%(harms[ff]*fmod,)] )
                sub1.axvline(x=1e-3*freq[ib0], linewidth=2, linestyle='-', color='k')
                sub1.axvline(x=harms[ff]*1e-3*fmod, linewidth=1, linestyle='--', color='k')
                sub1.axvline(x=1e-3*freq[ib1], linewidth=2, linestyle='-', color='k')
                sub2.axvline(x=1e-3*freq[ib0], linewidth=2, linestyle='-', color='k')
                sub2.axvline(x=harms[ff]*1e-3*fmod, linewidth=1, linestyle='--', color='k')                
                sub2.axvline(x=1e-3*freq[ib1], linewidth=2, linestyle='-', color='k')                
                sub3.axvline(x=1e-3*freq[ib0], linewidth=2, linestyle='-', color='k')
                sub3.axvline(x=harms[ff]*1e-3*fmod, linewidth=1, linestyle='--', color='k')                
                sub3.axvline(x=1e-3*freq[ib1], linewidth=2, linestyle='-', color='k')
            # endif
        # end for
    # end loop
    # ---------------- #
    
# end loop    
Pxy = _np.average(Pxy, axis=0)
Pxx = _np.average(Pxx, axis=0)
Pyy = _np.average(Pyy, axis=0)

Cxy = _np.abs( Pxy.conjugate()*Pxy ) / (_np.abs(Pxx)*_np.abs(Pyy) )
Cxy = _np.sqrt(Cxy)
phxy = _np.arctan2(_np.imag(Pxy), _np.real(Pxy))
phxy[_np.where(phxy<2.7)] += _np.pi    
phxy[_np.where(phxy>2.7)] -= _np.pi    

# --------------- #
   
sub4.plot(ece_rf, phxy.reshape(nch,nharm), 'o')
sub4b.plot(ece_rf, _np.sqrt(Cxy.reshape(nch,nharm)), 'x')
sub4.text(72, 2.0, 
          '%i to %i Hz + Harmonics'%(int(freq[is0]),int(freq[is1])),
          fontsize=12)
sub4.set_xlim(56,75)
#sub4.plot(ch+1, phxy.reshape(nch,nharm), 'o')
#sub4b.plot(ch+1, _np.sqrt(Cxy.reshape(nch,nharm)), 'rx')
#sub4.text(_np.average(ch+1), 2.0, 
#          '%i to %i Hz + Harmonics'%(int(freq[is0]),int(freq[is1])),
#          fontsize=12)
#sub4.legend(legs)

sub5.plot(ece_rf, _np.log(_np.abs(Pxy.reshape(nch,nharm))), 'o')
sub5b.plot(ece_rf, _np.sqrt(Cxy.reshape(nch,nharm)), 'x')
sub5.text(72, 2.0, 
          '%i to %i Hz + Harmonics'%(int(freq[is0]),int(freq[is1])),
          fontsize=12)
sub5.set_xlim(56,75)

#_plt.figure()
#_plt.xlabel('r [cm]')
#_plt.ylabel('Cross Coherence')
#_plt.plot(_np.abs(freqs-68.0)*cmPerGHz, Cxy, 'o' )

#savefig(_os.path.join(datafolder,scantitl), ext='png', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)    
#savefig(_os.path.join(datafolder,scantitl), ext='eps', close=False, 
#        verbose=True, dotsperinch = 300, transparency = True)