#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:57:34 2018

@author: weir
"""

# ========================================================================== #



from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #


import matplotlib.pyplot as _plt
import numpy as _np
import os as _os
from pybaseutils import plt_utils as _pltut

saveit = _os.path.join( 'G:/','W7XPapers','ECE_ECRH_Workshop_2018','hjcece','ifchain_')

fontdic = {"size":18, "weight":"normal", "family":"sans-serif", "sans-serif":"Arial"}
_plt.rc('font', **fontdic)
#_plt.rc('font', size=18, name='Arial')
    
fig_size = _plt.rcParams["figure.figsize"]
fig_size_spec = (2.875,3.35)
fig_size = (0.65*fig_size[0],0.65*fig_size[1])

#RF8 = _np.loadtxt("CECE_RFfreq_8GHz.txt", dtype=_np.float64, skiprows=1)  # +40 dB vid gain, -30 dBm input
IF4 = _np.loadtxt("CECE_IFfreq_4GHz.txt", dtype=_np.float64, skiprows=1)  # +40 dB vid gain, -30 dBm input
IF8 = _np.loadtxt("CECE_IFfreq_8GHz.txt", dtype=_np.float64, skiprows=1)  # +40 dB vid gain, -30 dBm input
IF12 = _np.loadtxt("CECE_IFfreq_12GHz.txt", dtype=_np.float64, skiprows=1) # +40 dB vid gain, -26 dBm input
IF16 = _np.loadtxt("CECE_IFfreq_16GHz.txt", dtype=_np.float64, skiprows=1) # +40 dB vid gain, -24 dBm input

IF4[:,1] *= -1.0
IF8[:,1] *= -1.0
IF12[:,1] *= -1.0
IF16[:,1] *= -1.0

#_plt.figure(figsize=fig_size)
#_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
#_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
#_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
#_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')
#
#_plt.xlabel(r'\Delta f [GHz]', fontsize=18, fontname='Arial')
#_plt.ylabel(r'V_{out} [VDC]', fontsize=18, fontname='Arial')
#_plt.title(r'IF Chain Measurements', fontsize=18, fontname='Arial')

IF4[:,1] -= _np.min(IF4[:,1])
IF8[:,1] -= _np.min(IF8[:,1])
IF12[:,1] -= _np.min(IF12[:,1])
IF16[:,1] -= _np.min(IF16[:,1])

IF4[:,1]  = 10*_np.log10(IF4[:,1])
IF8[:,1]  = 10*_np.log10(IF8[:,1])
IF12[:,1] = 10*_np.log10(IF12[:,1]) - 4
IF16[:,1] = 10*_np.log10(IF16[:,1]) - 6

IF4[:,1] = 10**(IF4[:,1]/10.0)
IF8[:,1] = 10**(IF8[:,1]/10.0)
IF12[:,1] = 10**(IF12[:,1]/10.0)
IF16[:,1] = 10**(IF16[:,1]/10.0)

hfig1 = _plt.figure(figsize=fig_size)
_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')

_plt.xlabel(r'$\Delta$ f [GHz]', fontsize=18, fontname='Arial')
_plt.ylabel(r'V$_{out}$ [VDC]', fontsize=18, fontname='Arial')
_plt.title(r'CECE-IF Back-end: -30 dBm input', fontsize=18, fontname='Arial')
_plt.xlim((-0.3, 0.3))

_plt.text(-0.285, 4.0, r'4 GHz IF', color='m', fontsize=18, fontname='Arial')
_plt.text(-0.285, 3.5, r'8 GHz IF', color='g', fontsize=18, fontname='Arial')
_plt.text(-0.285, 1.7, r'12 GHz IF', color='r', fontsize=18, fontname='Arial')
_plt.text(-0.285, 0.9, r'16 GHz IF', color='b', fontsize=18, fontname='Arial')
hfig1.tight_layout()

IF4[:,1] /= _np.max(IF4[:,1])
IF8[:,1] /= _np.max(IF8[:,1])
IF12[:,1] /= _np.max(IF12[:,1])
IF16[:,1] /= _np.max(IF16[:,1])

#_plt.figure()
#_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
#_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
#_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
#_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')
#
#_plt.xlabel(r'\Delta f [GHz]', fontsize=18, fontname='Arial')
#_plt.ylabel(r'V_{out} [a.u.]', fontsize=18, fontname='Arial')
#_plt.title(r'IF Chain Measurements: Normalized', fontsize=18, fontname='Arial')

IF4[:,1] = 10*_np.log10(IF4[:,1])
IF8[:,1] = 10*_np.log10(IF8[:,1])
IF12[:,1] = 10*_np.log10(IF12[:,1])
IF16[:,1] = 10*_np.log10(IF16[:,1])

fnew = _np.linspace(-0.5, 0.5, num=200)

#_plt.figure()
#_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
#_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
#_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
#_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')
#
#_plt.xlabel(r'\Delta f [GHz]', fontsize=18, fontname='Arial')
#_plt.ylabel(r'V_{out} [dB]', fontsize=18, fontname='Arial')
#_plt.title(r'IF Chain Measurements', fontsize=18, fontname='Arial')


hfig2 = _plt.figure(figsize=fig_size)
_plt.plot(fnew, _np.interp(fnew, IF4[:,0]-4.0, IF4[:,1]), 'm-')
_plt.plot(fnew, _np.interp(fnew, IF8[:,0]-8.0, IF8[:,1]), 'g-')
_plt.plot(fnew, _np.interp(fnew, IF12[:,0]-12.0, IF12[:,1]), 'r-')
_plt.plot(fnew, _np.interp(fnew, IF16[:,0]-16.0, IF16[:,1]), 'b-')

#_plt.figure()
#_plt.plot(fnew, _np.interp(fnew, IF4[:,0]-4.0, IF4[:,1])-_np.max(IF4[:,1]), 'm-')
#_plt.plot(fnew, _np.interp(fnew, IF8[:,0]-8.0, IF8[:,1])-_np.max(IF8[:,1]), 'g-')
#_plt.plot(fnew, _np.interp(fnew, IF12[:,0]-12.0, IF12[:,1])-_np.max(IF12[:,1]), 'r-')
#_plt.plot(fnew, _np.interp(fnew, IF16[:,0]-16.0, IF16[:,1])-_np.max(IF16[:,1]), 'b-')


_plt.xlabel(r'$\Delta$ f [GHz]', fontsize=18, fontname='Arial')
_plt.ylabel(r'V$_{out}$ [dB]', fontsize=18, fontname='Arial')
#_plt.title(r'IF Chain Measurements', fontsize=18, fontname='Arial')
_plt.xlim((-0.21, 0.21))
_plt.ylim((-3.0, 0.25))

_plt.text(-0.205, -0.25, r'4 GHz IF', color='m', fontsize=18, fontname='Arial')
_plt.text(-0.205, -0.50, r'8 GHz IF', color='g', fontsize=18, fontname='Arial')
_plt.text(-0.205, -0.75, r'12 GHz IF', color='r', fontsize=18, fontname='Arial')
_plt.text(-0.205, -1.00, r'16 GHz IF', color='b', fontsize=18, fontname='Arial')
hfig2.tight_layout()

if saveit:
    hname = saveit+'_CECEIF_BackEnd' # hfig.get_label()
    _plt.figure(hfig1.number)
    _pltut.savefig( _os.path.join(saveit,hname), ext='png', close = False, dotsperinch=100, transparency=True)
    _pltut.savefig( _os.path.join(saveit,hname), ext='eps', close = False, dotsperinch=100, transparency=True)
    hname = saveit+'_CECEIF_BackEnd' # hfig.get_label()

    hname = saveit+'_CECEIF_BackEnd_dB' # hfig.get_label()
    _plt.figure(hfig2.number)
    _pltut.savefig( _os.path.join(saveit,hname), ext='png', close = False, dotsperinch=100, transparency=True)
    _pltut.savefig( _os.path.join(saveit,hname), ext='eps', close = False, dotsperinch=100, transparency=True)
# endif