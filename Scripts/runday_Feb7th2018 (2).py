#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:32:58 2018

@author: weir
"""


# ========= #

import numpy as _np
import matplotlib.pyplot as _plt

# ========= #


VCOv=_np.asarray([-0.600,-0.550,-0.500,-0.450,-0.400,-0.350,-0.300,-0.250,-0.220,-0.200,-0.150,-0.100,-0.050,-0.000, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500],dtype=_np.float64)
VCOf=_np.asarray([12.890,12.900,12.910,12.910,12.920,12.920,12.930,12.940,12.960,12.980,13.020,13.050,13.090,13.130,13.180,13.230,13.300,13.370,13.440,13.510,13.590,13.670,13.760,13.850],dtype=_np.float64)

shot  =_np.asarray([ 69883, 69884, 69885, 69886, 69887, 69888, 69889, 69890, 69892, 69893, 69894, 69895, 69896, 69897, 69898, 69899, 69900, 69901, 69902, 69903, 69904, 69913, 69914, 69915, 69916, 69917, 69918, 69919, 69920, 69921, 69922,   69923, 69924, 69925, 69926, 69927, 69928, 69929, 69930],dtype=_np.float64)
IFreal=_np.asarray([13.075,13.075,13.062,13.050,13.037,13.025,13.012,13.012,13.012,13.012,13.095,13.095,13.095,14.075,14.065,14.055,14.045,14.035,14.025,14.085,14.095,14.075,14.075,14.095,14.095,14.075,14.065,14.055,14.045,14.035,14.025, 14.0125,13.095,13.075,13.055,13.035,13.115,13.075,13.075],dtype=_np.float64)
t0 =_np.asarray([    0.230, 0.230, 0.230, 0.230, 0.240, 0.230, 0.235, 0.300, 0.230, 0.240, 0.230, 0.235, 0.235, 0.230, 0.230, 0.230, 0.230, 0.230, 0.227, 0.235,     0,     0, 0.226, 0.226, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250],dtype=_np.float64)
t1 =_np.asarray([    0.330, 0.330, 0.320, 0.340, 0.340, 0.340, 0.260, 0.340, 0.290, 0.270, 0.280, 0.285, 0.285, 0.280, 0.280, 0.280, 0.280, 0.280, 0.280, 0.270,     0,     0, 0.270, 0.270, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300],dtype=_np.float64)
t2 =_np.asarray([        0,     0,     0,     0,     0,     0,     0, 0.240, 0.290, 0.300,     0,     0,     0, 0.300, 0.300, 0.300, 0.300, 0.300, 0.290, 0.270,     0,     0, 0.275, 0.275, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300],dtype=_np.float64)
t3 =_np.asarray([        0,     0,     0,     0,     0,     0,     0, 0.260, 0.340, 0.340,     0,     0,     0, 0.345, 0.345, 0.345, 0.345, 0.345, 0.350, 0.315,     0,     0, 0.330, 0.330, 0.350, 0.350, 0.350, 0.350, 0.350, 0.350],dtype=_np.float64)
#de =    [ 0.300, 0.300, 0.250, 0.200, 0.150, 0.100, 0.050, 0.050]

Vreal  =_np.asarray([ 0.000,-0.050,-0.100,-0.100,-0.150,-0.150,-0.200,-0.200,-0.200,-0.200, 0.050, 0.050, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.000, 0.100, 0.100, 0.150, 0.150, 0.150, 0.200, 0.250, 0.250, 0.300, 0.050,-0.050,-0.100,-0.150,-0.200,-0.250,-0.300, 0.350, 0.400],dtype=_np.float64)
IF1real=_np.asarray([13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000,13.000],dtype=_np.float64)
IF2real=_np.interp(Vreal, VCOv, VCOf)
#dr     = [ 0.190, 0.190,]

# ========= #

IF=4;
RFfixed = 56+IF
REFfixed=26.2

#de = _np.asarray(de, dtype=_np.float64)
#de = _np.atleast_1d(de)
#RFswept = RFfixed+de
def getECEIF(RFswept):
    IFswept = (RFswept-8.0)/4.0
    return IFswept
def getECERF(IFswept):
    IFswept = _np.asarray(IFswept)
    IFswept = _np.atleast_1d(IFswept)
    RFswept=4.0*IFswept+8
    return RFswept, RFswept-RFfixed
RFreal, realde = getECERF(IFreal)
IFswept = getECEIF(RFfixed)


#dr = _np.asarray(dr, dtype=_np.float64)
#dr = _np.atleast_1d(dr)
#REFswept= REFfixed+dr
#def getREFLIF(REFfixed, dr=0):
#    IFrefl1=(REFfixed-0.200)/2.0
#    IFrefl2=(REFfixed+dr-0.130)/2.0
#    return IFrefl1, IFrefl2

def getREFswept(IFreal2):
    IFreal2 = _np.asarray(IFreal2)
    IFreal2 = _np.atleast_1d(IFreal2)
    REFswept = 2.0*IFreal2+0.130
    return REFswept
def getREFfixed(IFreal1):
    IFreal1 = _np.asarray(IFreal1)
    IFreal1 = _np.atleast_1d(IFreal1)
    REFfixed = 2.0*IFreal1+0.200
    return REFfixed
def getREFLRF(IFrefl1, IFrefl2, dr=0):
    REFfixed = getREFfixed(IFrefl1)
    REFswept = getREFswept(IFrefl2)-dr
    return REFfixed, REFswept, REFswept-REFfixed
REFfreal, REFsreal, realdr = getREFLRF(IF1real, IF2real)
#IFrefl1, IFrefl2 = getREFLIF(REFfixed, dr)

#print('ECE settings (E-band): CECE-RF=%.3f, CECE-IF=%0.3f'%(RFswept[-1], RFfixed))
#print('         set (IF-band): CECE-RF=%.3f, CECE-IF=%0.3f'%(IFswept[-1], IF))
#print(['%0.3f'%(ie,) for ie in de])
#print(['%0.3f'%(ie,) for ie in IFreal])
#print(['%0.3f'%(ie,) for ie in RFswept])

#print('REFL settings (Ka-band): REFL-RF=%.3f, REFL-IF=%0.3f'%(REFswept[-1], REFfixed))
#print('          set (IF-band): REFL-RF=%.3f, REFL-IF=%0.3f'%(IFrefl2[-1], IFrefl1))
##print('%0.3f'%(dr,))
#print(['%0.3f'%(ir,) for ir in dr])
#print(['%0.3f'%(ir,) for ir in REFswept])


FIX11 = shot[:12]
FIX12 = shot[12:20]
FIX21 = shot[20:32]
FIX22 = shot[32:]

_plt.figure()
_plt.subplot(2,2,1)
_plt.plot(shot[:13]-shot[0]+1,RFreal[:13]-60,'bo')
_plt.plot(shot[13:20]-shot[0]+1,RFreal[13:20]-64,'bx')
_plt.plot(shot[20:32]-shot[0]+1,RFreal[20:32]-64,'rx')
_plt.plot(shot[32:]-shot[0]+1,RFreal[32:]-60,'ro')
#_plt.plot(shot-shot[0]+1,(RFreal[:12]-60).tolist()
#                        +(RFreal[12:20]-64).tolist()
#                        +(RFreal[20:]-64).tolist(),'o')
#_plt.plot(FIX11-FIX11[0]+1,RFreal[:12]-60,'go')
#_plt.plot(FIX12-FIX12[0]+1,RFreal[12:]-64,'bo')
_plt.subplot(2,2,2)
_plt.plot(IFreal[:13],RFreal[:13]-60,'bo')
_plt.plot(IFreal[13:20]-1.0, RFreal[13:20]-64,'bx')
_plt.plot(IFreal[20:32]-1.0, RFreal[20:32]-64,'rx')
_plt.plot(IFreal[32:], RFreal[32:]-60,'ro')

_plt.subplot(2,2,3)
_plt.plot(shot[:20]-shot[0]+1,realdr[:20],'bo') #, RFswept)
_plt.plot(shot[20:]-shot[0]+1,realdr[20:],'ro') #, RFswept)

_plt.subplot(2,2,4)
_plt.plot(Vreal[:20], REFsreal[:20]-REFfreal[:20], 'bo')
_plt.plot(Vreal[20:], REFsreal[20:]-REFfreal[20:], 'ro')
_plt.plot(VCOv,getREFswept(VCOf)-REFfixed, '.-') #, REFswept)
