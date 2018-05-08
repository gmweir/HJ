#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:57:34 2018

@author: weir
"""

import matplotlib.pyplot as _plt
import numpy as _np


IF4 = _np.loadtxt("CECE_IFfreq_4GHz.txt", dtype=_np.float64, skiprows=1)
IF8 = _np.loadtxt("CECE_IFfreq_8GHz.txt", dtype=_np.float64, skiprows=1)
IF12 = _np.loadtxt("CECE_IFfreq_12GHz.txt", dtype=_np.float64, skiprows=1)
IF16 = _np.loadtxt("CECE_IFfreq_16GHz.txt", dtype=_np.float64, skiprows=1)

IF4[:,1] *= -1.0
IF8[:,1] *= -1.0
IF12[:,1] *= -1.0
IF16[:,1] *= -1.0

_plt.figure()
_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')

IF4[:,1] -= _np.min(IF4[:,1])
IF8[:,1] -= _np.min(IF8[:,1])
IF12[:,1] -= _np.min(IF12[:,1])
IF16[:,1] -= _np.min(IF16[:,1])

_plt.figure()
_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')

IF4[:,1] /= _np.max(IF4[:,1])
IF8[:,1] /= _np.max(IF8[:,1])
IF12[:,1] /= _np.max(IF12[:,1])
IF16[:,1] /= _np.max(IF16[:,1])

_plt.figure()
_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')

IF4[:,1] = 10*_np.log10(IF4[:,1])
IF8[:,1] = 10*_np.log10(IF8[:,1])
IF12[:,1] = 10*_np.log10(IF12[:,1])
IF16[:,1] = 10*_np.log10(IF16[:,1])

fnew = _np.linspace(-0.5, 0.5, num=200)

_plt.figure()
_plt.plot(IF4[:,0]-4.0, IF4[:,1], 'm.-')
_plt.plot(IF8[:,0]-8.0, IF8[:,1], 'g.-')
_plt.plot(IF12[:,0]-12.0, IF12[:,1], 'r.-')
_plt.plot(IF16[:,0]-16.0, IF16[:,1], 'b.-')

_plt.figure()
_plt.plot(fnew, _np.interp(fnew, IF4[:,0]-4.0, IF4[:,1]), 'm-')
_plt.plot(fnew, _np.interp(fnew, IF8[:,0]-8.0, IF8[:,1]), 'g-')
_plt.plot(fnew, _np.interp(fnew, IF12[:,0]-12.0, IF12[:,1]), 'r-')
_plt.plot(fnew, _np.interp(fnew, IF16[:,0]-16.0, IF16[:,1]), 'b-')

#_plt.figure()
#_plt.plot(fnew, _np.interp(fnew, IF4[:,0]-4.0, IF4[:,1])-_np.max(IF4[:,1]), 'm-')
#_plt.plot(fnew, _np.interp(fnew, IF8[:,0]-8.0, IF8[:,1])-_np.max(IF8[:,1]), 'g-')
#_plt.plot(fnew, _np.interp(fnew, IF12[:,0]-12.0, IF12[:,1])-_np.max(IF12[:,1]), 'r-')
#_plt.plot(fnew, _np.interp(fnew, IF16[:,0]-16.0, IF16[:,1])-_np.max(IF16[:,1]), 'b-')


_plt.xlabel(r'Frequency [GHz]')
_plt.xlabel('Frequency [GHz]')
_plt.ylabel('Voltage [VDC]')
_plt.title('IF Response')