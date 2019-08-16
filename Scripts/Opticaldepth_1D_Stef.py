# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:25:39 2019

@author: s146959
"""
def frequencybw(freqc,numpts=1,freqbw=None): #Create multiple points around central value as finite bandwith of ECE
    if freqbw is None:
        freqbw = 0.2*_np.ones_like(freqc)
    freqc=_np.atleast_1d(freqc)
    numch = len(freqc)
#    numpts = 100
    freqL = freqc-freqbw/2
    freqH = freqc+freqbw/2
    
    if numpts>1:
        freq = _np.zeros( (numpts, numch), dtype=float)
        for gg in range(numch):  # gg = 1:numch
            freq[:,gg] = _np.linspace(freqL[gg], freqH[gg], num=numpts, endpoint=True)
        # end for
        return freq
    else:
        return freqc


import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
from FIT.model_spec import model_qparab
from opticaldepth_X2_1D import opticaldepth_X2_1D
from pybaseutils import utils as _ut

datafolder = _os.path.abspath(_os.path.join('..','..','..','..', 'SyntheticDiagnostic','test'))#location of MCviewer Ray Trace output

interpfactor=100 #factor af the amount of extra points in interpolation
numpts = 50 #number of points in finite bandwidth of ECE
freqc=(64.5,68,70) #central frequencies of the ECE

#Obtain parameters from file
[rho,x,y,z,modB,T] = _np.loadtxt(datafolder, dtype=float, unpack=True, 
                        usecols=(1,3,4,5,12,13), skiprows=(6))
x = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), x, 
                        None, _np.arange(interpfactor*len(rho)-1))
y = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), y, 
                        None, _np.arange(interpfactor*len(rho)-1))
z = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), z, 
                        None, _np.arange(interpfactor*len(rho)-1))
modB = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), modB, 
                        None, _np.arange(interpfactor*len(rho)-1))
T = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), T, 
                        None, _np.arange(interpfactor*len(rho)-1))
rho = _ut.interp(_np.arange(0,len(rho)*interpfactor,interpfactor), rho, 
                        None, _np.arange(interpfactor*len(rho)-1))

RD=_np.sqrt(x**2+y**2) #calculate R from x and y

#plot temperature profile obtained from MCviewer aslong the ECE ray
_plt.figure()
_plt.plot(rho,T)
_plt.xlabel(r'$\rho$') 
_plt.ylabel('T [keV]')
_plt.title('Temperature profile MCviewer along ray')

#make linear grid for profile calculation
roa=_np.linspace(0,1,100)

#load profile parameters
qparab_parameters_NITB=[0.051783,0.002,10.208,3.4433,0.72003,0.5]
qparab_parameters_TITB=[2.1728,0.034019,6.4493,18.432,-0.7886,0.12922]
qparab_parameters_N=[0.0831,0.001,8.4211,2.1603,0.8,0.49132]
qparab_parameters_T=[1.0728,0.087646,2.8536,5.1638,-0.4393,0.16118]

#calculate density and temperature profiles based on linear grid and profile parameters
Nprofile=model_qparab(roa,qparab_parameters_N)
Tprofile=model_qparab(roa,qparab_parameters_T)
NITBprofile=model_qparab(roa,qparab_parameters_NITB)
TITBprofile=model_qparab(roa,qparab_parameters_TITB)

#prevent negative values (found at the end of density profiles)
Nprofile[0][Nprofile[0]<0]=0 
NITBprofile[0][NITBprofile[0]<0]=0

#plot temperature and density profiles made by qparab model
_plt.figure()
_plt.plot(roa,Tprofile[0])
_plt.plot(roa,TITBprofile[0])
_plt.xlabel(r'$\rho$') 
_plt.ylabel('T [keV]')
_plt.title('Temperature profiles qparab along plasma roa')

_plt.figure()
_plt.plot(roa,Nprofile[0])
_plt.plot(roa,NITBprofile[0])
_plt.xlabel(r'$\rho$') 
_plt.ylabel('n [10^19 parts/m^3]')
_plt.title('Density profiles qparab along plasma roa')

#initialize tau en kImk for saving data
tau=_np.zeros((len(freqc),numpts+1))
kImk=_np.zeros((len(freqc),len(rho),numpts+1))

#run opticaldepth script for every number of points
for numpts in range(1,numpts+1):
    freq=frequencybw(freqc,numpts) #calculate used frequency points
    [tau[:,numpts],info]=opticaldepth_X2_1D(Te=_np.array(Tprofile[0]), 
                    ne_in=_np.array(Nprofile[0]*10**14),
                    roa_T=_np.atleast_1d(roa), roa_n=_np.atleast_1d(roa),
                    rho=rho,modB=modB,RD=RD,ZD=z,freq=_np.atleast_2d((freq))
                    ,plotit=False)
    #density should be multiplied to 10^20 and converted to cm^3 so to 10^-6
    kImk[:,:,numpts]=info.Imk

#plot kImk and tau as a function of number of points    
_plt.close("all")
_plt.plot(kImk[0,:,1:])
_plt.xlabel('Im(k)') 
_plt.title('Imaginary part of k')
_plt.figure()
_plt.plot(range(1,numpts),tau[0][1:numpts])
_plt.xlabel('number of points') 
_plt.ylabel(r'$\tau$')
_plt.title('Optical depth')
