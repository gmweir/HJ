# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:45:51 2017

@author: gawe
"""

import numpy as _np
import os as _os
import matplotlib.pyplot as _plt
from mpl_toolkits import mplot3d
from pybaseutils import plt_utils as _pltut

wd = _os.path.abspath(_os.path.curdir)
coilfil = _os.path.join(wd,'coils.h-j')

if _os.path.exists(coilfil):

    dat = _np.loadtxt(coilfil, skiprows=3, unpack=False)

    nfilaments = sum(dat[:,3] == 0)
    ncoils = 8
    ifilaments = _np.where(dat[:,3]==0)[0]
    ifilaments = _np.concatenate((_np.atleast_1d(-1), ifilaments), axis=0)

    # ====== #
    fig = _plt.figure()
#    ax = _plt.gca()
    ax = fig.add_subplot(111, projection='3d')

    # ====== #

    for ii in range(nfilaments):
        istart = ifilaments[ii]+1
        iend = ifilaments[ii+1]+1
        npts = iend-istart

        if ii < 24:
            jj = ii
            if jj == 0:
                HC = _np.zeros([npts,3,24], dtype=float)   # 4 x npts x 8
            # endif
            HC[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(HC[:,0,jj], HC[:,1,jj], HC[:,2,jj], '-', color=_pltut.rgb('Red'))

        elif ii < 26:
            jj = ii-24
            if jj == 0:
                OV1 = _np.zeros([npts,3,2], dtype=float)   # 4 x npts x 2
            # endif
            OV1[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(OV1[:,0,jj], OV1[:,1,jj], OV1[:,2,jj], '-', color=_pltut.rgb('LightBlue'))

        elif ii < 28:
            jj = ii-26
            if jj == 0:
                OV2 = _np.zeros([npts,3,2], dtype=float)   # 4 x npts x 2
            # endif
            OV2[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(OV2[:,0,jj], OV2[:,1,jj], OV2[:,2,jj], '-', color=_pltut.rgb('Magenta'))

        elif ii < 30:
            jj = ii-28
            if jj == 0:
                OV3 = _np.zeros([npts,3,2], dtype=float)   # 4 x npts x 2
            # endif
            OV3[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(OV3[:,0,jj], OV3[:,1,jj], OV3[:,2,jj], '-', color=_pltut.rgb('Orange') )

        elif ii < 38:
            jj = ii-30
            if jj == 0:
                TA = _np.zeros([npts,3,16], dtype=float)   # 4 x npts x 16
            # endif
            TA[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(TA[:,0,jj], TA[:,1,jj], TA[:,2,jj], '-', color=_pltut.rgb('Blue'))

        elif ii < 46:
            jj = ii-38
            if jj == 0:
                TB = _np.zeros([npts,3,16], dtype=float)   # 4 x npts x 16
            # endif

            TB[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(TB[:,0,jj], TB[:,1,jj], TB[:,2,jj], '-', color=_pltut.rgb('DarkGreen'))

        elif ii < 48:
            jj = ii-46
            if jj == 0:
                AV = _np.zeros([npts,3,2], dtype=float)   # 4 x npts x 2
            # endif
            AV[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(AV[:,0,jj], AV[:,1,jj], AV[:,2,jj], '-', color=_pltut.rgb('Purple') )

        elif ii < 50:
            jj = ii-48
            if jj == 0:
                IV = _np.zeros([npts,3,2], dtype=float)   # 4 x npts x 2
            # endif
            IV[:,:,jj] = dat[istart:iend,:3]
            ax.plot3D(IV[:,0,jj], IV[:,1,jj], IV[:,2,jj], '-', color=_pltut.rgb('Cyan') )

    #end for

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Heliotron J Coils')
# end if




hdr = 'periods  4 \n'
hdr += 'begin filament \n'
hdr += 'mirror NUL \n'

# Helical coil model
R0=1.2
AC=0.22
alpha=-0.40
ph = 0.0
ph -= _np.pi/4
#ph += 3.5*_np.pi/180.0
tt = _np.linspace(0, 1, 201)
xt = (1.2 + AC*_np.cos(2*_np.pi*4*tt))*_np.cos(2*_np.pi*tt+ph)
yt = (1.2 + AC*_np.cos(2*_np.pi*4*tt))*_np.sin(2*_np.pi*tt+ph)
zt = AC*_np.sin(2*_np.pi*4*tt)

#fig2 = _plt.figure();  ax2 = fig2.add_subplot(111, projection='3d')
ax.plot3D(xt, yt, zt, 'b-', lw=1)


# === #
HCavg = _np.mean(HC, axis=2)
xt1 = HCavg[:,0]
yt1 = HCavg[:,1]
zt1 = HCavg[:,2]
#xt1 = xt + 0.43*AC*_np.cos(2*_np.pi*tt+ph)
#yt1 = yt + 0.43*AC*_np.sin(2*_np.pi*tt+ph)
#zt1 = zt + 0.43*AC*_np.sin(2*_np.pi*4*tt)
ax.plot3D(xt1, yt1, zt1, 'b-', lw=1)
