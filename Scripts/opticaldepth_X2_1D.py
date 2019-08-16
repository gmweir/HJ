
# ========================================================================== #    
# ========================================================================== #    

from __future__ import absolute_import, with_statement, absolute_import, \
                        division, print_function, unicode_literals

# ========================================================================== #    
# ========================================================================== #    

from pybaseutils import Struct
from pybaseutils import utils as _ut
from pybaseutils import plt_utils as _pltut
import numpy as _np
import matplotlib.pyplot as _plt

# ========================================================================== #    
# ========================================================================== #    


def opticaldepth_X2_1D(Te=None, ne_in=None, roa_T=_np.atleast_1d(0.1), roa_n=_np.atleast_1d(0.1), freq=None, 
                       modB=_np.linspace(2.90, 2.30, 201), RD=_np.linspace(6.0, 5.25, 201), ZD=_np.linspace(0.1, -0.1, 201), rho=_np.linspace(-1, 1, 201), 
                       beamwidth=0.025, plotit=True, info=None, nargout=2):
    """
     function [tau,info] = opticaldepth_X2_1D(Te,ne_in,roa_T,roa_n,freq,modB,RD,ZD,rho,beamwidth,plotit,info)
    
    
     note: Doesn't work for half-tesla. Optical depth is for 2nd harm x-mode.
       This code calculates the resonance locations of each of the ECE
       channels and estimates the absorption profiles / optical depth by using
       analytic functions from (1).  It also uses modB to determine the
       cyclotron frequency profile.
    
       Optical depth is the integral of the absorption over the ray-trajectory
       (multiplied by two) and hence each ray has a different optical depth.
       It is not a function of effective radius, and it IS a function of
       frequency.  Each ECE channel has it's own optical depth.
    
    Inputs:
      Te              - [KeV],Temperature
      ne_in           - [parts/cm3], density
      roa_T           - [-], Effective radius locations for temperature measurement
      roa_n           - [-], Effective radius locations for density
      measurement
      freq            - [GHz], vector of frequencies for which to calculate absorption
      modB,RD,ZD,rho  - [T,m,m,-], domain of magnetic field, major radius,
                           height above axis, and effective radius
      plotit   - do you want to make the plots?   Boolean
    
    Outputs:  All in effective radius (flux coords.) rho = sqrt(Psi/Psi_lcfs)
    
    
    (1)Alikaev, V.V., Litvak, A.G., Suvorov, E.V., Friman, A.A., High-frequency
    plasma heating, edited by Litvak, A. G., New York: American Institute of
    Physics, 1992, pp.3-14
    """

    # tic
    
    #If there is no input.  Give 'em something to work with.
    if info is None:
        info = Struct()
    # end if
    
    if hasattr(info, 'Rant'):
        Rantenna = info.Rant
    else:
        Rantenna = 6.5
    # endif
    if hasattr(info, 'Zant'):
        Zantenna = info.Zant
    else:
        Zantenna = 0.15
    # endif
    
    if Rantenna<100:    Rantenna *= 100    # endif
    if Zantenna<100:    Zantenna *= 100    # endif
    
    if Te is None:      
        t_fit = 1.17*(0.05+(1.0-0.05)*(1-_np.abs(rho)**2)**15)   
#        t_fit = 1.17*(0.05+(1.0-0.05)*(1.0-_np.abs(rho)**1.75)**5)   
#        t_fit = 7.17*(0.05+(1.0-0.05)*(1.0-_np.abs(rho)**1.75)**5)   
    else:
        edge_temp = 0.050 #[KeV]

        #Remove NaN's from linear fitting section
        nanT = _np.isnan(Te) + _np.isnan(roa_T)
        Te = Te[~nanT] 
        roa_T = roa_T[~nanT]

        if (_np.max(roa_T)<1.0):          #Ensure temperature extends to edge
            Te    = _np.append(Te, _np.min([edge_temp,Te[-1]]))
            roa_T = _np.append(roa_T, 1.0)
        # endif

        if (_np.min(roa_T)>0.0):          #Ensure temperature gradient @ core = 0
            Te = _np.hstack((_np.flipud(Te), Te))
            roa_T = _np.hstack((-_np.flipud(roa_T),roa_T))
        # endif

        t_fit = _ut.interp(roa_T, Te, None, _np.abs(rho)) #[KeV]            
   # endif
        
    if ne_in is None:
        n_fit = 6e18
#        n_fit = 0.3e20     # m-3
        n_fit = 1e-6*n_fit # cm-3
        n_fit *= (1.0-_np.abs(rho)**1.5)**1.5                      
#        n_fit *= (1.0-_np.abs(rho)**2.)**.25                      
#        n_fit *= (0.05+(1.0-0.05)*(1.0-_np.abs(rho)**10)**5)   
        
    else:
        edge_dens = 0.5e12   #[10^12 cm-3]
    
        #Remove NaN's from linear fitting section
        nanN = _np.isnan(ne_in)+_np.isnan(roa_n)
        ne_in = ne_in[~nanN] 
        roa_n = roa_n[~nanN]
    
        if (_np.max(roa_n)<1.0):          #Ensure density extends to edge
            ne_in = _np.append(ne_in, _np.min([edge_dens,ne_in[-1]]))
            roa_n = _np.append(roa_n, 1.0)
        # endif

        if (_np.min(roa_n)>0.0):          #Ensure density gradient @ core = 0
            ne_in = _np.hstack((_np.flipud(ne_in), ne_in))
            roa_n = _np.hstack((-_np.flipud(roa_n), roa_n))
        # endif

        n_fit = _ut.interp(roa_n, ne_in, None, _np.abs(rho)) #[10^12 cm-3]
    # endif
    
#    colors=['b','r','g','k','m','b','c','m']
    
    #Plot the profiles used to determine the optical depth to figure 210.
    if plotit:
        _plt.figure(210)
        _plt.subplot(2,1,1)
        _plt.plot(rho,t_fit,'r-')
        _plt.grid('on')
#        _plt.xlabel(r'r/a$_p$')
        _plt.axis((-0.2, 1, 0, 1.2*_np.max(t_fit)))
        _plt.ylabel(r'T$_e$ [keV]')
        _plt.title('Electron Temperature Profile')
    
        _plt.subplot(2,1,2)
        _plt.plot(rho,n_fit,'b-')
        _plt.grid('on')
        _plt.xlabel(r'r/a$_p$')
        _plt.axis((-0.2, 1.0, 0.0, 1.2*_np.max(n_fit)))
        _plt.ylabel(r'n$_e$ [cm$^{-3}$]')
        _plt.title('Plasma Density Profile')
    # endif
    
    # =========================================================================== #
    
    if freq is None:
        #ECRH frequency of the Varian gyrotrons
#        freqc  = _np.atleast_1d(140.0) #[GHz],
#        freqbw = _np.atleast_1d(0.05)  #[GHz],
    
        freqc = _np.linspace(135, 160, 13)    
        freqbw = 0.05*_np.ones_like(freqc)
    
        numch = len(freqc)
        numpts = 1
        freqL = freqc-freqbw/2
        freqH = freqc+freqbw/2
    
        if numpts>1:
            freq = _np.zeros( (numpts, numch), dtype=float)
            for gg in range(numch):  # gg = 1:numch
                freq[:,gg] = _np.linspace(freqL[gg], freqH[gg], num=numpts, endpoint=True)
            # end for
        else:
            freq = freqc
        # endif
    # endif
    
    if freq.any()>1e9:
        freq *= 1e-9
    # endif
    if len(_np.shape(freq)) == 1:
        numpts=1
    elif len(_np.shape(freq)) == 2 and _np.shape(freq)[0] == 1:
        numpts=1
        numch=len(freq)
   
#    sizfreq = _np.shape(freq)
#    numch = sizfreq[1]
#    numpts = sizfreq[0]
    if _np.shape(freq)[0] == 1 and numpts>1:  freq = freq.T  # endif
    freq = _np.atleast_2d(freq)
    sizfreq = _np.shape(freq)
    numch = sizfreq[1]
    numpts = sizfreq[0]
    
    if nargout>1:
    #     info.freqc = freqc
    #     info.freqbw = freqbw
        info.freq = freq
    # endif
    
    #Preallocate all terms and arrays
    freq_vec = freq.reshape( (numpts*numch,1), order='F')  #[nfreq,1],Reshape the array for fast computation
    nfreq    = len(freq_vec)    #Number of loops below
    nR       = len(RD)
    
    #Absorption and the other terms are local quantities
    kImk = _np.ones( (nfreq, nR), dtype=float)
#    sizImk = _np.size(kImk)
    
    # =========================================================================== #
    
    #Use the beam-path to estimate the step-size for integration:
    if _np.max(RD)<1e2:    RD = 1e2*RD  # endif
    if _np.max(ZD)<1e2:    ZD = 1e2*ZD  # endif      #[cm], 1xnR
    dR = _np.diff(RD)
    dZ = _np.diff(ZD)
    dRp = _np.sqrt( _np.nanmean(dR)**2.0+_np.nanmean(dZ)**2.0 )  #step size
    
    #Edit the grid for resonance locating.  Remove Inf's and NaN's
    remall = _np.isnan(rho)+_np.isinf(rho)+(rho<-1.0)+(rho>1.0)+_np.isnan(RD)+_np.isnan(ZD)+_np.isnan(t_fit)
    real_rho = rho.copy()
    real_rho[remall] = 0.0 #[1xnR]
    real_R = RD.copy()
    real_R[remall] = 0.0 #[1xnR]
    real_Z = ZD.copy()
    real_Z[remall] = 0.0 #[1xnR]
    
    real_rho = _np.dot( _np.ones( (numch,1), dtype=float), real_rho.reshape((1,nR)))         #[nfreq,nR]
    real_R = _np.dot( _np.ones( (numch,1), dtype=float), real_R.reshape((1,nR)))           #[nfreq,nR]
    real_Z = _np.dot( _np.ones( (numch,1), dtype=float), real_Z.reshape((1,nR)))           #[nfreq,nR]
    
    # =========================================================================== #
    
    #Calculate the optical depth for 2nd harmonic x-mode
    me = 9.1e-28
    c = 3e10
    qe = 4.8e-10 #some fundamental constansts, cgs.
    
    wce = 1.0e4*qe/me/c*modB         #[1xnR] of cyclotron frequencies,
    wce = wce.reshape((1,nR))
    #Convert Tesla to Gauss, modB =10^4*modB
    beta2 = (3.2e-9/(me*c**2.0))*t_fit #[1xnR] Normalized thermal velocity^2 / c^2.
    beta2 = beta2.reshape((1,nR))
    #Convert te from KeV to ergs -> te=1.6e-9*te
    wp = _np.sqrt(4.0*_np.pi*qe**2.0/me*n_fit)   #[1xnR] Plasma frequency, n_fit ~ 10^12 parts/cm3
    wp = wp.reshape((1,nR))
    
    omega = 2e9*_np.pi*freq_vec             #[1xnfreq] freq is given in GHz -> Hz -> rad/s
    omega = omega.reshape((nfreq,1), order='F')
    
    # =========================================================================== #
    
    #Convert to matrices of appropriate dimensions and values
    omega = _np.dot(omega, _np.ones((1,nR), dtype=float))   #[nfreq,nR]
    omegadiv = 1.0/omega                 #[nfreq,nR], for fast computation do now
    wce   = _np.dot(_np.ones((nfreq,1), dtype=float), wce)   #[nfreq,nR]
    beta2 = _np.dot(_np.ones((nfreq,1), dtype=float), beta2) #[nfreq,nR]
    beta2div = 1.0/beta2                 #[nfreq,nR], for fast computation do now
    wp    = _np.dot(_np.ones((nfreq,1), dtype=float), wp)    #[nfreq,nR]
    
    # =========================================================================== #
    
    q = (2.0*omegadiv*wp)**2.0   #[nfreq,nR], Density factor ~ c^2/Alfven velocity^2 pg7
    gamma2 = ((3.0-0.5*q)/(3.0-q))**2.0  #[nfreq,nR], parameter in the optical depth eqn.
    
    u2 = (2.0*omegadiv*wce)**2.0  #[nfreq,nR], eqn 23 parameter    (2*wce/omega)^2
    u2rtdiv = 1.0/_np.sqrt(u2)    #[nfreq,nR]
    
    #TYPO IN BOOK? IT IS DIVIDED BY BETA2 NOT BETA
    # zet = 2*(1-u2rtdiv ).*sqrt(beta2div)  #[nfreq,nR*nZ], z parameter in ref. eqn 28
    # zet = 2*(s*wH-w)/(s*wH*beta) = 2*(1-w/(2wH))/(beta)= 2*(1-u2rtdiv)/sqrt(beta2)
    zet = 2.0*(1.0-u2rtdiv)*beta2div  #[nfreq,nR*nZ], z parameter in ref. eqn 23
    
    # Filter rejects values outside bounds of resonance condition
    indres = (0.0<zet)*(zet<64.0)
    zet[~indres] = 0.0
    
    #Calculate local absorption [cm-1]:  2nd harm X-mode eqn 28 pg 13
    kImk =  (omega/c)*(2.0*_np.sqrt(_np.pi)/15.0)*q*gamma2*(zet**2.5)*_np.exp(-zet)
    kImk[_np.isnan(kImk)] = 0.0      #If NaN set = 0
    kImk[kImk<0.0] = 0.0      #If unphysical set = 0
    
    # =========================================================================== #
    
    newsiz = (sizfreq[0], sizfreq[1], nR)
    kImk = kImk.reshape(newsiz, order='F') 
    if numpts > 1:
        vImk   = _np.squeeze( _np.nanvar(kImk,axis=0) )
   
        df     = freq[1, ...]-freq[0,...]
        freqbw = freq[-1,...]-freq[0,...]
        freqc  = _np.nanmean(freq, axis=0)
        kImk = _np.squeeze(_np.sum(kImk,axis=0))
#        kImk = _np.squeeze(_np.trapz(kImk.squeeze(), x=freq, axis=0))        
        kImk = _np.atleast_2d(kImk)
#        kImk   = _np.dot((df/(freqbw+df)).reshape(numch,1), kImk)
#        kImk   = _np.dot((1.0/(freqbw)).reshape(numch,1), kImk)
        kImk   = kImk*_np.dot((df/(freqbw+df)).reshape(numch,1), _np.ones((1,nR), dtype=float))
#        kImk   = kImk*_np.dot((df/(freqbw)).reshape(numch,1), _np.ones((1,nR), dtype=float))
#        kImk   = kImk*_np.dot((1.0*_np.ones((numch,),dtype=float)/(numpts)).reshape(numch,1), _np.ones((1,nR), dtype=float)) #is equivalent
    else:
        freqc = freq
        kImk = _np.squeeze(kImk)
        vImk = _np.zeros_like(kImk)
    # endif
    coeff_intensity = _np.dot(((2.0*_np.pi*1e9*freqc)**2.0/(8.0*_np.pi**3.0*(1e-2*c)**2.0)).reshape(numch,1), 
                              _np.ones((1,len(RD[~remall])), dtype=float))

    if len(_np.shape(kImk)) == 1: kImk=_np.atleast_2d(kImk)
    info.absorption = 2.0*kImk[..., ~remall]    
    info.integrated_absorption = _np.fliplr(_np.cumsum(_np.fliplr(info.absorption), axis=1)*dRp)
    # info.integrated_absorption = cumsum(info.absorption,2)*dRp
    tau = (info.integrated_absorption[...,0]).T       #[nch,numpts] for most cases
    
    if nargout>1:
        # if plotit
        dtau2 = tau**2.0*( _np.sum(vImk.reshape(numch,nR)**2.0/numpts**2.0, axis=1).T )
        # # endif
        info.tau = tau
    
    #     beamwidth = 1e2*beamwidth
    #     Z_rp = inline('H_mc-(R_mc-R).*tan(2*rotang-pi/2).T)
    #     Zax  = 1e2*Z_rp(.1,1e-2*RD,1.4454+0.29,0.95)
        #This is a function that expresses the power density of the ECE beam
    #     beamwidth = beamwidth(8)
    #     if size(beamwidth,1)>1
    #     if length(beamwidth)>1 #assumes you are passing a vector of beamwidths for each channel individually
    #         Pdens = 2.*exp( -2.*( ones(numch,1)*(ZD-Zax)./cos(0.95) )**2./(beamwidth.T*ones(1,nR)./2)**2 ) ...
    #             ./ ( pi .* (beamwidth.T*ones(1,nR)./2)**2 )
    #     else
    #         Pdens = 2.*exp( -2.*( (ZD-Zax)./cos(0.95) )**2./(beamwidth./2)**2 ) ...
    #             ./ ( pi .* (beamwidth./2)**2 )
    #         Pdens(isnan(Pdens)) = 0.0
    #         Pdens = ones(numch,1)*Pdens
    #     # endif
        Pdens = _np.ones( _np.shape(kImk), dtype=float) #no power density weighting
    
        # ====================================================================== #
    
        #Some parameters needed for calculated weightings and variances:
    #     totP   = trapz( Pdens(:,~remall).*(dRp) ,2)
        totP   = 1.0
        maxAbs = _np.dot(_np.max(kImk, axis=1).reshape(numch,1), _np.ones((1,nR), dtype=float)) #Maximum absorption at each freq
        nImk   = kImk/maxAbs #normalize the absorption by it's maximum
        tokImk = _np.dot(_np.sum(kImk*Pdens*dRp, axis=1).reshape(numch,1), _np.ones((1,nR), dtype=float)) #Total absorption for each frequency
        nImks  = kImk*Pdens/tokImk #normalize the absorption by it's sum
    
        # ====================================================================== #
    
        info.emissivity = coeff_intensity*_np.dot(_np.ones((numch,1), dtype=float), _np.atleast_2d(t_fit[~remall]))
        info.emissivity *= info.absorption*_np.exp(-info.integrated_absorption)
        info.integrated_emissivity = _np.fliplr(_np.cumsum(_np.fliplr(info.emissivity), axis=1)*dRp)
        # info.integrated_emissivity = cumsum(info.emissivity,2)*dRp
    
        #First-pass emission
        info.Tfirstpass = (_np.trapz(Pdens[:,~remall]*dRp*info.emissivity/coeff_intensity,axis=1)/totP).T
        info.spatavg = (_np.trapz(Pdens[:,~remall]*dRp*info.absorption*_np.exp(-info.integrated_absorption),axis=1)/totP).T
    
        #Absorption weighted electon temperature (measured by radiometry):
    #     info.Tinterp = Tweight./info.spatavg
    
    #     info.Tfirstpass = info.Tinterp.*(1-exp(-tau))
        info.Tavg = info.Tfirstpass/info.spatavg
        
        # ====================================================================== #
    
        #Absorption weighted position:
        roa_Imk = real_rho*nImks
        R_Imk   = real_R  *nImks
        Z_Imk   = real_Z  *nImks
    
        #These are the absorption weighted averaged positions:
        ece_roa = _np.nansum(roa_Imk*dRp, axis=1).T #[nfreq,1], in roa
        ece_R   = _np.nansum(R_Imk*dRp  , axis=1).T #[nfreq,1]  in R
        ece_Z   = _np.nansum(Z_Imk*dRp  , axis=1).T #[nfreq,1]  in Z
    
        #     varroa = _np.nanvar( roa_Imk, axis=1).T #The width of absorption line in roa
        #     varR   = _np.nanvar( R_Imk  , axis=1).T #The width of absorption line in R
        #     varZ   = _np.nanvar( Z_Imk  , axis=1).T #The width of absorption line in Z
    
        #Full-width at half-max may be a more appropriate definition.  Non-gaussian:
        fullwidth_rho = _np.zeros( _np.shape(ece_roa), dtype=float)
        fullwidth_Z = fullwidth_rho.copy()
        fullwidth_R = fullwidth_rho.copy()
        nImk[ _np.abs(nImk)<0.5 ] = 0.0
        for gg in range(numch):  # gg = 1:numch
            pos = _np.where(nImk[gg,:])[0]             #Find all positions within the full-width
            fullwidth_rho[gg] =  _np.max( rho[pos] ) - _np.min( rho[pos] )
            fullwidth_R[gg]   =  _np.max(  RD[pos] ) - _np.min(  RD[pos] )
            fullwidth_Z[gg]   =  _np.max(  ZD[pos] ) - _np.min(  ZD[pos] )
        # end for
    
    #         stdroa = _np.sqrt( varroa )
    #         stdR   = _np.sqrt( varR )
    #         stdZ   = _np.sqrt( varZ )
    
        # ====================================================================== #
        if 1 and plotit:
           info.ecepath = _np.sqrt( (RD[~remall]-Rantenna)**2.0+(ZD[~remall]-Zantenna)**2.0)
           _plt.figure()
           _plt.xlabel(r'r/a$_p$', fontsize=14, fontname='Arial')
           _plt.ylabel(r'T$_e$ [eV]', fontsize=14, fontname='Arial')
           _plt.ylabel(r'First-pass Radiation Temperature', fontsize=14, fontname='Arial')
           chlfs = ece_roa >= 0.0
           chhfs = ece_roa <  0.0
           _plt.plot( ece_roa[chlfs], 1e3*info.Tfirstpass[chlfs],'-', lw=2, color='b')
           _plt.plot(-ece_roa[chhfs], 1e3*info.Tfirstpass[chhfs],'-',lw=2, color=_pltut.rgb('DarkGreen'))
           _plt.plot( rho[rho>0.0], 1e3*t_fit[rho>0.0],'-', lw=2, color='k')
    
           _plt.figure()
           _plt.xlabel(r'Path [cm]', fontsize=14, fontname='Arial')
           _plt.ylabel(r'Absorption [cm$^{-1}$]', fontsize=14, fontname='Arial')
           _plt.plot( info.ecepath, info.absorption.T,'-', lw=2)
           xlims = _plt.xlim()
           ylims = _plt.ylim()
           _plt.text(xlims[0]+(xlims[1]-xlims[0])*0.1, ylims[0]+(ylims[1]-ylims[0])*0.8,r'$\alpha_\omega(s)$', color='k')
    
           _plt.figure()
           _plt.xlabel(r'Path [cm]', fontsize=14, fontname='Arial')
           _plt.ylabel(r'Optical Depth', fontsize=14, fontname='Arial')           
           _plt.plot(info.ecepath, info.integrated_absorption.T, '-', lw=2)
           xlims = _plt.xlim()
           ylims = _plt.ylim()
           _plt.text(xlims[0]+(xlims[1]-xlims[0])*0.1, ylims[0]+(ylims[1]-ylims[0])*0.8, r'$\tau=\int\alpha_\omega(s)ds$', color='k')
    
           _plt.figure()
           _plt.xlabel(r'Path [cm]', fontsize=14, fontname='Arial')
           _plt.ylabel(r'Emissivity [a.u.]', fontsize=14, fontname='Arial')           
           _plt.plot( info.ecepath, info.emissivity.T,'-',lw=2)
    #        _plt.plot( R[~remall],info.emissivity/max(info.emissivity.flatten()),'-',lw=2)
           xlims = _plt.xlim()
           ylims = _plt.ylim()
           _plt.text(xlims[0]+(xlims[1]-xlims[0])*0.1, ylims[0]+(ylims[1]-ylims[0])*0.8, r'j$_\omega(s)$',color='k')
    
           _plt.figure()
           _plt.xlabel(r'Path [cm]', fontsize=14, fontname='Arial')
           _plt.ylabel(r'Integrated Emissivity [a.u.]', fontsize=14, fontname='Arial')           
           _plt.plot(info.ecepath, info.integrated_emissivity.T,'-',lw=2)
           xlims = _plt.xlim()
           ylims = _plt.ylim()
           _plt.text(xlims[0]+(xlims[1]-xlims[0])*0.1, ylims[0]+(ylims[1]-ylims[0])*0.8, r'$\int j_\omega(s)ds$',color='k')    
        # endif
        # ====================================================================== #
    
        info.rho     = rho
        info.RD      = RD
        info.ZD      = ZD
        info.Imk     = kImk
        info.tau     = tau
        info.stdtau  = _np.sqrt(dtau2)
        info.ece_roa = ece_roa
        info.ece_R   = ece_R
        info.ece_Z   = ece_Z
    #         info.stdroa  = stdroa
        info.roa_fwhm = fullwidth_rho
    #         info.stdR    = stdR
        info.R_fwhm   = fullwidth_R
    #         info.stdZ    = stdZ
        info.Z_fwhm   = fullwidth_Z
    # endif
    if plotit:
        PD = _np.sqrt((RD-Rantenna)**2.0+(ZD-Zantenna)**2.0)
        _plt.figure()
        _plt.plot(PD, kImk.T,'-'),
        y_max = 1.05*_np.max( _np.max( kImk.flatten() ) )
        _plt.xlabel('Path [cm]') 
        _plt.ylabel(r'Imk$_j$ (cm$^{-1}$)')
        _plt.grid('on')
        _plt.axis(((_np.min(PD)-1.0), (_np.max(PD)+1.0), 0.0, 1.2*_np.max((y_max,3))))
        _plt.title('Local Absorption Shapes')
    
        _plt.figure()
        ece_D = _np.sqrt( (ece_R-Rantenna)**2.0+(ece_Z-Zantenna)**2.0 )
        D_lower = _np.sqrt(fullwidth_R**2.0+fullwidth_Z**2.0)/2.0
        D_upper = _np.sqrt(fullwidth_R**2.0+fullwidth_Z**2.0)/2.0
        _plt.errorbar(ece_D, tau, yerr=_np.sqrt(dtau2), xerr=[D_lower, D_upper], fmt='mo') #plot(ece_R,tau,'mo'),
        _plt.xlabel('Path [cm]') 
        _plt.ylabel(r'$\tau$')
        _plt.grid('on')
#        _plt.ylim((0,5))
        _plt.title('Optical Depth')
    
        _plt.figure()
        _plt.subplot(1,2,1) 
        _plt.grid('on')
        freqc = (_np.max(freq, axis=0)+_np.min(freq, axis=0))/2.0
        freqbw = (_np.max(freq, axis=0)-_np.min(freq, axis=0))
    
        #Plot ECE res. location in configuration space vs frequency-space.
        _plt.errorbar(freqc,_np.abs(ece_roa),yerr=fullwidth_rho/2.0,xerr=freqbw/2.0, fmt='bo')
        _plt.xlabel('Frequency (GHz)') 
        _plt.ylabel(r'r/a$_p$')
        _plt.title('ECE channels') #axis([50 62 0 1])
    
        _plt.subplot(1,2,2)
        _plt.grid('on')
        _plt.errorbar(freqc, tau, yerr=_np.sqrt(dtau2), xerr=freqbw/2.0, fmt='bo')
        _plt.xlabel('frequency [GHz]') 
        _plt.ylabel(r'$\tau$')
        _plt.title('Optical Depth')
    # endif
    
    # toc
    return tau, info
# end def

if __name__=="__main__":
    tau, info = opticaldepth_X2_1D()
# end if
    
    
    
    