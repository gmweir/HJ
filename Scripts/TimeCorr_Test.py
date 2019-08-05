# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:09:56 2019

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

from FFT.fft_analysis import fftanal, ccf
from pybaseutils.plt_utils import savefig
from FFT.windows import _crosscorr


def chunk(x, n):
    '''Split the list, xs, into n chunks'''
    L = len(x)
    assert 0 < n <= L
    s = L//n
    return [x[p:p+s] for p in range(0, L, s)]

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'Workshop'))
#datafolder = _os.path.join('/homea','weir','bin')
print(datafolder)

#
#tt=_np.linspace(0,2.0*_np.pi,401)
#tt2=_np.linspace(0,4.0*_np.pi,801)
#sin1=_np.sin(tt)
#sin2=_np.sin(tt)
#
##sin1=_np.array([0,0.5,1,2,3,3,4,2,1,0.5,0])
##sin2=_np.array([0,0.5,1,2,3,3,4,2,1,0.5,0])
##tt=range(len(sin1))
#
#x=_crosscorr(sin1,sin2)/_np.sum(sin1**2)
#[t,y]=ccf(sin1,sin2,1/((tt[-1]-tt[0])/(len(sin1)-1)))
#fig0=_plt.figure()
#_plt.plot(sin1)
#_plt.plot(sin2)
#fig=_plt.figure()
#_plt.plot(y)
#fig.suptitle('ccf method on sines')
#fig1=_plt.figure()
#_plt.plot(x)
#fig1.suptitle('crosscorr method on sines')



cmPerGHz = 1
for nwindows in [1,10,100,1000,10000]:
#    nwindows = 100
    overlap=0.0
    sintest=True
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
    
    intb = [15e3, 500e3]  # original
    #intb = [50e3, 400e3]  # broadband fluctuations
    # intb = [400e3, 465e3]  # high frequency mode
    
    tb=[0.29,0.31]
    #tb=[0.3,0.39]
    #tb = [0.192, 0.370]
    
    nfils = len(fils)
    
    #for ii in range(1):
    #    
    #    filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
    #    print(filn)
    #    tt, tmpRF, tmpIF = \
    #        _np.loadtxt(filn, dtype=_np.float64, unpack=True, usecols=(0,1,2))
    #    tt = 1e-3*tt    
    #    tt_tb=[_np.where(tt==tb[0])[0][0],_np.where(tt==tb[1])[0][0]]
    #    [tau,co]=ccf(tmpRF,tmpIF,(len(tt[tt_tb[0]:tt_tb[1]])-1)/(tt[tt_tb[1]]-tt[tt_tb[0]]))
    #    fig=_plt.figure()
    #    sub1=_plt.subplot(3,1,1)
    #    sub2=_plt.subplot(3,1,2,sharex=sub1)
    #    sub3=_plt.subplot(3,1,3)
    #    
    #    sub1.plot(tt[tt_tb[0]:tt_tb[1]],tmpRF[tt_tb[0]:tt_tb[1]])
    #    sub2.plot(tt[tt_tb[0]:tt_tb[1]],tmpIF[tt_tb[0]:tt_tb[1]])
    #    sub3.plot(tau,co)
        
    for ii in range(1):
        
        if not sintest:
            filn = _os.path.abspath(_os.path.join(datafolder, fils[ii]))
            print(filn)
            tt, tmpRF, tmpIF = \
                _np.loadtxt(filn, dtype=_np.float64, unpack=True, usecols=(0,1,2))
            tt = 1e-3*tt
            tt_tb=[_np.where(tt<=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
            tt_used=tt[tt_tb[0]:tt_tb[1]]
            RF_used=tmpRF[tt_tb[0]:tt_tb[1]]
            IF_used=tmpIF[tt_tb[0]:tt_tb[1]]
            
        if sintest:
            tb=[0,2.0]
            df=15.50e3
            _np.random.seed()
            n_s=5000001
            tt=_np.linspace(0,2.0*_np.pi,n_s)
            fs=1/(((tt[len(tt)-1]-tt[0])/len(tt)))
            RF_used=0.05*_np.sin(2.0*_np.pi*(df)*tt)
            ppp=fs/df #points per period
            delay=20/fs #no of points delay/frequency = time delay
            IF_used=0.02*_np.sin(2.0*_np.pi*(df)*(tt-delay))+_np.random.standard_normal( (tt.shape[0],) )
            tt_tb=[_np.where(tt<=tb[0])[0][0],_np.where(tt>=tb[1])[0][0]]
            tt_used=tt[tt_tb[0]:tt_tb[1]]
    
        if tt_used[1]-tt_used[0]!=tt_used[2]-tt_used[1]:
            tt2=_np.linspace(tt_used[0],tt_used[-1],len(tt_used),endpoint=True)
            tmpRF=_np.interp(_np.asarray(tt2,dtype=float),tt,tmpRF)
            tmpIF=_np.interp(_np.asarray(tt2,dtype=float),tt,tmpIF)
            tt_used=tt2     
            
#        if sintest:
#            tt_used=_np.linspace(0,2.0*_np.pi,40001)
#            RF_used=_np.sin(tt_used)
#            IF_used=_np.sin(tt_used)
        
        Navr=nwindows
        nsig=len(tt_used)
        windowoverlap=overlap
        nwins = int(_np.floor(nsig/(Navr-Navr*windowoverlap + windowoverlap)))
        noverlap = windowoverlap*nwins
        ist = _np.arange(Navr)*(nwins - noverlap)
        ist=ist.astype(int)
        
        
        tau_total=0
        co_total=0
        cxy_total=0
        z_total=0
        for gg in range(Navr):
            istart=ist[gg]
            iend=istart+nwins
            freq=len(tt_used[istart:iend])/(tt_used[iend-1]-tt_used[istart])
            [tau,co]=ccf(RF_used[istart:iend],IF_used[istart:iend],freq)
            co_total+=co
            
            cxy=_crosscorr(RF_used[istart:iend],IF_used[istart:iend])
            cxy=cxy/_np.sum(RF_used[istart:iend]**2)
            #/(_np.sum(RF_used[istart:iend]**2.0)*_np.sum(IF_used[istart:iend]**2.0))
            cxy_total+=cxy
            
            z=0.5*_np.log(_np.abs((1+cxy)/(1-cxy)))
            z_total+=z
            
        avco=co_total/Navr
        avcxy=cxy_total/Navr
        avz=z_total/Navr
        avcxy_z=(_np.exp(2.0*avz)-1)/(_np.exp(2.0*avz)+1)
    
        lags=_np.arange(0,len(RF_used[istart:iend]))
    #    lags=_np.arange(0,512)
        taucrosscorr=lags/float(freq)
       
    #    fig2=_plt.figure()
    #    _plt.plot(tau,_np.abs(avco))
    #    fig2.suptitle('ccf method averaged')
    
    #    fig3=_plt.figure()
    #    _plt.plot(taucrosscorr,_np.abs(avcxy))
    #    fig3.suptitle('crosscorr method averaged')
    ##    _plt.plot(tau,avcxy_z)
    #    
        fig4=_plt.figure()
        _plt.plot(tau,(avco))
#        _plt.plot(taucrosscorr,(avcxy)) 
    #    _plt.plot(taucrosscorr,_np.abs(avcxy_z)) #equal as avcxy
        fig4.suptitle('cff (blue) and crosscorr (orange) method averaged correlation')
    
    Bvid=0.5e6
    Bif=200e6        
    sqrtNs = _np.sqrt(2*Bvid*(tb[-1]-tb[0]))
    sens = _np.sqrt(2*Bvid/Bif/sqrtNs)
    print('$\tau$= '+str(tau[_np.where(avco==max(avco))[0][0]]))
    if sintest:
        print('delay= '+str(delay))
# end for



# ========================================================================== #
# ========================================================================== #

# ========================================================================== #
# ========================================================================== #
    
    
    
    
    
    