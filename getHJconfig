#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:46:17 2017

@author: weir
"""
# ========================================================================== #

#This section is to improve compatability between bython 2.7 and python 3x
from __future__ import absolute_import, division, print_function, unicode_literals
__metaclass__ = type

# ========================================================================== #

import numpy as _np
import json as _jsn
import os as _os
import matplotlib.pyplot as _plt
from collections import OrderedDict as _orddict
import time as _time
#import matplotlib.pyplot as _plt
#import datetime as _dt

from .qmejson import utc2time, parseProgramid, getTimeIntervalfromID
from .fetch import pickECEfile
from pybaseutils import utils as _ut 

# ========================================================================== #

# 'CECE-BackEnd_170117-65609.json'
# calib = _os.path.join(curdir, 'HJPY', 'Calib-files')
# 
# class ECE_frontend(_ut.Struct):
#     curdir = _os.path.abspath(_os.curdir)    
#     frontend = _os.path.join(curdir, 'HJPY', 'Front-End-files')
#     
#     def __init__(self, cfgfilname=None, verbose=True):
#         """
#         Initialize the variables, if no configuration filename is given.
#         """      

# ====================================================================== #


# Sub class with information about the specific front end used    
class frontend(_ut.Struct()):
    def __init__(self, fe_name=None, SubBand=0, **kwargs):
        self.frontend = fe_name
        self.SubBand = SubBand
        if fe_name is None:
            self.Remark = str()  # notes about the front-end settings
            self.PORT = _np.float64()  # [#], HJ port #
            
            # Cartesian coordinates of antenna origin and target, [m, m, m]
            self.Antenna_CartCoords_m = _np.asarray([0,0,0], dtype=_np.floaf64)
            self.Target_CartCoords_m = _np.asarray([0,0,0], dtype=_np.floaf64)

            # [m], focal length of antenna connected to front end              
            self.Antenna_FocLength_m = _np.asarray([0, 0], dtype=_np.floaf64)
            
            # RF band information
            self.SubBand = SubBand  # subband of the corresponding front end (indice)
            self.LO = _np.float64()  # [GHz], front end local oscillator frequecy
            self.SIDEBAND = str()  # [str], Upper or lower sideband of radiometer            
            
            self.FrontEnd_RFAtt_dB = _np.float64(0.0)  # [dB], RF front end attenuation
            self.FrontEnd_RFGain_dB = _np.float64(0.0)  # [dB], RF front end gain   

            # Lower and upper IF bandwidth of the RF amplifiers in the front-end
            self.IFbandwidth = _np.asarray([0.0, 0.0], dtype=_np.float64)  # [GHz]
        else:
            # super(frontend, self).__dict2vars__(kwargs)
       # end if
    # end def __init__    
# end class frontend definitions

        
# ====================================================================== #

        
class frontend_config(_ut.Struct):
    curdir = _os.path.abspath(_os.curdir)    
    frontend = _os.path.join(curdir, 'HJPY', 'Front-End-files')
    
    def __init__(self, filname=None, frontend=None, subband=0, verbose=True):
        """
        Initialize the variables, if no configuration filename is given.
        """
        self.verbose = verbose
        if filname is None:
            self.filename = str()  # configuration file name

            self.User = str()     # person who wrote the file
            self.Valid_Since = str()  # valid date for data
            self.Modified_At = str()  # date last modified
            self.Active = [str()]  # the active front end used by the ECE
            self.General_Remarks = str()           
        else:   
            self.filename = filname            
            self.readECEconfig(filname)
    # end def
            
    # ====================================================================== #
    # Define some extra magic methods.
    
    # ===== Define length returning the number of channels and the multiplication which returns the factors necessary for the calibration.
    def __len__(self):
        return self.nch

    # ===== Define greater than, less than, and equal based on valid date of the parameter file.
    def __eq__(self, other):
        return (self.filename == other)
    # end def magic method equal, ==  
    
    def __lt__(self, other):
        return (self.dateval < other)
    # end def magic method less than, <  
        
    def __gt__(self, other):
        return (self.dateval > other)
    # end def magic method greater than, >          

    # ====================================================================== #
    
    def _feobj_(self, **kwargs):
        fe = _ut.Struct()
        fe.PORT = kwargs.get('PORT', _np.float64(0.0))  # [#], HJ port #
        fe.AntennaCoords = kwargs.get('Antenna_CartCoords_m', list())  # [m,m,m], cartesian coord's of antenna
        fe.TargetCoords = kwargs.get('Target_CartCoords_m', list())  # [m,m,m], cartesian coordinates of antenna target
        fe.Focus = kwargs.get('Antenna_Foc_length_m', list())  # [m], focal length of antenna connected to front end        
        fe.remark = kwargs.get('Remark', str())  # note about the channel  

        fe.LO = kwargs.get('LO', _np.float64(0.0))  # [GHz], front end local oscillator frequecy
        fe.SIDEBAND = kwargs.get('SIDEBAND', str())  # [str], Upper or lower sideband of radiometer
        fe.IFBandwidth = kwargs.get('IFBandwidth', list())  # [GHz], lower and upper IF bandwidth (amplifiers)
        fe.frontend_att = kwargs.get('FrontEnd_RFAtt_dB', _np.float64(0.0))  # [dB], RF front end attenuation
        fe.frontend_gain = kwargs.get('FrontEnd_RFGain_dB', _np.float64(0.0))  # [dB], RF front end gain            
        return fe
        
    # ====================================================================== #
    
                
    def readECEconfig(self, filname):
        """
        filname has to be the path pointing to the file.

        The function assigns the values loaded from the .json file to specific values.
        """
        
        # ===== Load the .json file containing the configuration file.
        with open( filname, 'r') as fid:         
            config = _jsn.load( fid )                
        # end file open section
         
        # ===== Assign the strings.
        self.user = config['User']
        self.dateval = config['Valid_Since']
        self.lastmod = config['Modified_At']
        self.remarks = config['General_Remarks']
        self.filename = filname
                
        # ===== Initialize the list for the channel specific information.
        self.ADC = list()
        self.BAND = list()
        self.IF = list()
        self.BW = list()
        self.RF = list()
        self.DCoffset = list()
        self.GAIN = list()
        self.coupling = list()
        self.HPF3dB = list()
        self.LPF3dB = list()
        self.remark = list()

        # ===== Assign/Append the channel specific information.
        for ii in range(self.nch):  #0:15
            ch = 'CH'+str(ii+1).zfill(2)
#            self.ch[ii] = self._chobj_(config['Channels'][ch]) 
            self.ADC.append(config['Channels'][ch]['ADC_CH_NAME'])  # DAQ channel name
            self.BAND.append(config['Channels'][ch]['BAND'])  # Branch indicator
            self.IF.append(config['Channels'][ch]['IF'])  # [GHz], ECE intermediate frequency
            self.BW.append(config['Channels'][ch]['BW'])  # [GHz], ECE bandwidth
            self.RF.append(config['Channels'][ch]['Frequency'])  # [GHz], ECE microwave frequecy
            self.DCoffset.append(config['Channels'][ch]['DCoffset_VDC'])  # [VDC], DC offset voltage      
            self.GAIN.append(config['Channels'][ch]['Amplification_VV']) #[V/V], Post-detection amplifier gain        
            self.coupling.append(config['Channels'][ch]['Coupling'])  # AC / DC 
            self.HPF3dB.append(config['Channels'][ch]['HPF_Frequency_KHz'])  # [KHz], high-pass filter 3 dB bandwidth)
            self.LPF3dB.append(config['Channels'][ch]['LPF_Frequency_KHz'])  # [KHz], low-pass filter 3 dB bandwidth)            
            self.remark.append(config['Channels'][ch]['Remark'])       
        # end for 
        self.BAND = _np.asarray(self.BAND, dtype=_np.int64)
        self.IF = _np.asarray(self.IF, dtype=_np.float64)
        self.BW = _np.asarray(self.BW, dtype=_np.float64)
        self.RF = _np.asarray(self.RF, dtype=_np.float64)
        self.DCoffset = _np.asarray(self.DCoffset, dtype=_np.float64)
        self.GAIN = _np.asarray(self.GAIN, dtype=_np.float64)
        self.HPF3dB = _np.asarray(self.HPF3dB, dtype=_np.float64)
        self.LPF3dB = _np.asarray(self.LPF3dB, dtype=_np.float64)
    # enddef readECEconfig     

    # ====================================================================== #

    def setECEconfig( self, configset, configval ):
        """
        configset chooses the entry in the dictionary, that configval should be saved to.
        """
        self.AllNames[configset] = configval                
    # end def 

    def saveECEconfig(self, validsince, sfilename=None):
        """
        sfilename specifies the filename in which the configuration data from the self.AllNames dictionary should be written.
        validsince specifies the date, since when the configuration file will be valid. This string has to be in the format yymmdd-HHMM
        
        The resulting .json file is written to the corresponding folder in the x-drive with a properly formatted filename.
        Note that the version number is automatically chosen, so that it is always larger than the previously used one, which should be more save for erroneously saved files.
        """
        
        FrontEnd_Name = 'CECE-RF'
        
        # ====== Build the filename. Increase version number, until no file with the chosen version number is found. Save this version number in self.AllNames['Version'].
        if sfilename is None:
            sfilename = 'CECE-FrontEnd_'+ validsince
            sfilename = _os.path.join( self.backend, sfilename + '.json' )            
#            ii = 1
#            while _os.path.exists(sfilename) and ( ii < 99 ):          
#                ii = ii+1
#                sfilename = 'QME-configuration_'+ validsince +'_v' + str(ii)
#                self.AllNames['Version'] = ii
#                sfilename = _os.path.join( backend, sfilename + '.json' )
#            # endwhile
        else:
            validsince = sfilename[14:25]
        # endif          
            
        # ===== Initialize Config_All_Data, which will be a subdictionary of the later on defined ECE_Config_All_Data that contains the channel specific data.

        FrontEnd_Data = dict()
        for jj in ['PORT', 'Active', 'LO', 'SIDEBAND', 'IFBandwidth',
                   'FrontEnd_RFAtt_dB', 'FrontEnd_RFGain_dB', 
                   'Antenna_CartCoords_m', 'Target_CartCoords_m'
                   'Antenna_FocLength_m', 'Remark']:
            FrontEnd_Data[FrontEnd_Name][jj].setdefault(self.AllNames[jj+' '+FrontEnd_Name])                    
        # end for

        # ===== Initialize ECE_Config_All_Data, which is a dictionary that contains all information that will be written to the parameter .json file.
        ECE_Config_All_Data = _orddict([
            ("User", self.AllNames['User'] ),
            ("Valid_Since", self.AllNames['Valid_Since'] ),
            ("Modified_At", str( utc2time( _time.time()*1e9 ) ) ),
            ("General_Remarks", self.AllNames['General_Remarks'] ),
            (FrontEnd_Name, FrontEnd_Data)
                    ])
        
        # ==== Dump ECE_Config_All_Data to a .json file with the filename as evaluated above.
        with open( sfilename, 'w') as fid:
            _jsn.dump(ECE_Config_All_Data, fid, sort_keys=False, indent=4, ensure_ascii=False)        
        #fileopen
             
        # ======  Print success message to the command line.
        if self.verbose:
            print('Successfully saved file '+sfilename) 
        # endif
    # end def
    
    # ====================================================================== #
        
# end class frontend_config

# ========================================================================== #


class backend_config(_ut.Struct):
    curdir = _os.path.abspath(_os.curdir)    
    backend = _os.path.join(curdir, 'HJPY', 'Back-End-files')
    
    def __init__(self, filname=None, verbose=True):
        """
        Initialize the variables, if no configuration filename is given.
        """
        self.verbose = verbose
        if filname is None :
            self.filename = str()  # configuration file name

            self.user = str()     # person who wrote the file
            self.dateval = str()  # valid date for data
            self.lastmod = str()  # date last modified
 
            self.frontend = list() # Corresponding front-end
            self.band = list()    # Bands in this device
            self.rotatt = list()  # [dB], rotary attenuator setting
            self.fixatt = list()  # [dB], fixed attenuator setting
            self.Fs = _np.float64()  # [KHz], sampling frequency
            self.remarks = str() 

            self.nch = _np.int64() # number of DAQ channels
#            self.AllNames = dict()  # dictionary for saving
        else:   
            self.filename = filname            
            self.readECEconfig(filname)
            self.getECEfreq()
    # end def
        
#    def getECEfreq(self):
#        self.ece_freq = _np.asarray([ ch.RF for ch in self.ch ], dtype=_np.float64)
#        self.ece_bw = _np.asarray([ ch.BW for ch in self.ch ], dtype=_np.float64)                         
#        return self.ece_freq, self.ece_bw
        
    # ====================================================================== #
    # Define some extra magic methods.
    
    # ===== Define length returning the number of channels and the multiplication which returns the factors necessary for the calibration.
    def __len__(self):
        return self.nch

    def __mul__(self, other):
        nt = _np.size(other, axis=0)
        return _np.dot(_np.ones((nt,1), dtype=_np.float64), self.bit_to_volt[:, 0].reshape(1, self.nch))*other 

    # ===== Define greater than, less than, and equal based on valid date of the parameter file.
    def __eq__(self, other):
        return (self.filename == other)
    # end def magic method equal, ==  
    
    def __lt__(self, other):
        return (self.dateval < other)
    # end def magic method less than, <  
        
    def __gt__(self, other):
        return (self.dateval > other)
    # end def magic method greater than, >          

    # ====================================================================== #
    
    def _chobj_(self, **kwargs):
        ch = _ut.Struct()
        ch.ADC = kwargs.get('ADC_CH_NAME', str()) # DAQ channel name
        ch.BAND = kwargs.get('BAND', int(0))  # Branch indicator
        ch.IF = kwargs.get('IF', _np.float64(0.0))  # [GHz], ECE intermediate frequency
        ch.BW = kwargs.get('BW', _np.float64(0.0))  # [GHz], ECE bandwidth
        ch.RF = kwargs.get('Frequency', _np.float64(0.0))  # [GHz], ECE microwave frequecy
        ch.RFl = kwargs.get('Frequency_Min', _np.float64(0.0))  # [GHz], RF upper end
        ch.RFu = kwargs.get('Frequency_Max', _np.float64(0.0))  # [GHz], RF lower end        
        ch.backend_att = kwargs.get('BackEnd_RFAtt_dB', _np.float64(0.0))  # [dB], RF back end attenuation
        ch.backend_gain = kwargs.get('BackEnd_RFGain_dB', _np.float64(0.0))  # [dB], RF back end gain            
        ch.backend_vidgain = kwargs.get('VideoGain_dB', _np.float64(0.0))  # [dB], back end video gain
        ch.DCoffset = kwargs.get('DCoffset_VDC', _np.float64(0.0))  # [VDC], DC offset voltage      
        ch.GAIN = kwargs.get('Amplification_VV', _np.float64(1.0)) #[V/V], Post-detection amplifier gain        
        ch.coupling = kwargs.get('Coupling', str())  # AC / DC 
        ch.HPF3dB = kwargs.get('HPF_Frequency_KHz', _np.float64(0.0))  # [KHz], high-pass filter 3 dB bandwidth)
        ch.LPF3dB = kwargs.get('LPF_Frequency_KHz', _np.float64(0.0))  # [KHz], low-pass filter 3 dB bandwidth)
        ch.remark = kwargs.get('Remark', str())  # note about the channel  
        return ch     
    # ====================================================================== #
                
    def readECEconfig(self, filname):
        """
        filname has to be the path pointing to the file.

        The function assigns the values loaded from the .json file to specific values.
        """
        
        # ===== Load the .json file containing the configuration file.
        with open( filname, 'r') as fid:         
            config = _jsn.load( fid )                
        # end file open section
         
        # ===== Assign the strings.
        self.user = config['User']
        self.dateval = config['Valid_Since']
        self.lastmod = config['Modified_At']

        self.frontend = config['Frontend']
        self.remarks = config['General_Remarks']
        self.filename = filname

        # ===== Assign the integers.
        self.band = config['Band']
        if self.dateval<'2017-01-01':
            self.nch = 16
        else:
            self.nch = len( config['Channels'] )        
        # endif
        
        # ====== Assign the floats.
        self.rotatt = _np.float64(config['RotAttenuator_dB'])
        self.fixatt = _np.float64(config['FixedAttenuator_dB'])
        self.Fs = _np.float64(1.0e3*config['Sampling_Rate_KHz'])
        
        # ===== Initialize the list for the channel specific information.
        self.ADC = list()
        self.BAND = list()
        self.IF = list()
        self.BW = list()
        self.RF = list()
        self.DCoffset = list()
        self.GAIN = list()
        self.coupling = list()
        self.HPF3dB = list()
        self.LPF3dB = list()
        self.remark = list()

        # ===== Assign/Append the channel specific information.
        for ii in range(self.nch):  #0:15
            ch = 'CH'+str(ii+1).zfill(2)
#            self.ch[ii] = self._chobj_(config['Channels'][ch]) 
            self.ADC.append(config['Channels'][ch]['ADC_CH_NAME'])  # DAQ channel name
            self.BAND.append(config['Channels'][ch]['BAND'])  # Branch indicator
            self.IF.append(config['Channels'][ch]['IF'])  # [GHz], ECE intermediate frequency
            self.BW.append(config['Channels'][ch]['BW'])  # [GHz], ECE bandwidth
            self.RF.append(config['Channels'][ch]['Frequency'])  # [GHz], ECE microwave frequecy
            self.DCoffset.append(config['Channels'][ch]['DCoffset_VDC'])  # [VDC], DC offset voltage      
            self.GAIN.append(config['Channels'][ch]['Amplification_VV']) #[V/V], Post-detection amplifier gain        
            self.coupling.append(config['Channels'][ch]['Coupling'])  # AC / DC 
            self.HPF3dB.append(config['Channels'][ch]['HPF_Frequency_KHz'])  # [KHz], high-pass filter 3 dB bandwidth)
            self.LPF3dB.append(config['Channels'][ch]['LPF_Frequency_KHz'])  # [KHz], low-pass filter 3 dB bandwidth)            
            self.remark.append(config['Channels'][ch]['Remark'])       
        # end for 
        self.BAND = _np.asarray(self.BAND, dtype=_np.int64)
        self.IF = _np.asarray(self.IF, dtype=_np.float64)
        self.BW = _np.asarray(self.BW, dtype=_np.float64)
        self.RF = _np.asarray(self.RF, dtype=_np.float64)
        self.DCoffset = _np.asarray(self.DCoffset, dtype=_np.float64)
        self.GAIN = _np.asarray(self.GAIN, dtype=_np.float64)
        self.HPF3dB = _np.asarray(self.HPF3dB, dtype=_np.float64)
        self.LPF3dB = _np.asarray(self.LPF3dB, dtype=_np.float64)
    # enddef readECEconfig     

    # ====================================================================== #

    def setECEconfig( self, configset, configval ):
        """
        configset chooses the entry in the dictionary, that configval should be saved to.
        """
        self.AllNames[configset] = configval                
    # end def 

    def saveECEconfig( self, validsince, sfilename=None):
        """
        sfilename specifies the filename in which the configuration data from the self.AllNames dictionary should be written.
        validsince specifies the date, since when the configuration file will be valid. This string has to be in the format yymmdd-HHMM
        
        The resulting .json file is written to the corresponding folder in the x-drive with a properly formatted filename.
        Note that the version number is automatically chosen, so that it is always larger than the previously used one, which should be more save for erroneously saved files.
        """

        # ====== Build the filename. Increase version number, until no file with the chosen version number is found. Save this version number in self.AllNames['Version'].
        if sfilename is None:
            sfilename = 'CECE-BackEnd_'+ validsince
            sfilename = _os.path.join( self.backend, sfilename + '.json' )            
#            ii = 1
#            while _os.path.exists(sfilename) and ( ii < 99 ):          
#                ii = ii+1
#                sfilename = 'QME-configuration_'+ validsince +'_v' + str(ii)
#                self.AllNames['Version'] = ii
#                sfilename = _os.path.join( backend, sfilename + '.json' )
#            # endwhile
        else:
            validsince = sfilename[14:25]
        # endif          
            
        # ===== Initialize Config_All_Data, which will be a subdictionary of the later on defined ECE_Config_All_Data that contains the channel specific data.
        Channel_Data = dict()
        for ii in range(0, self.nch):
            channel = 'CH'+str(ii+1).zfill(2)
            for jj in ['ADC_CH_NAME', 'BAND', 'IF', 'BW', 'Frequency',
                       'Frequency_Max', 'Frequency_Min', 'BackEnd_RFAtt_dB',
                       'BackEnd_RFGain_dB', 'VideoGain_dB', 'DCOffset_VDC',
                       'Amplification_VV', 'Coupling', 'HPF_Frequency_KHz',
                       'LPF_Frequency_KHz', 'Remark']:
                Channel_Data[channel][jj].setdefault(self.AllNames[jj+' '+channel])                    
            # end for
        # end for

        # ===== Initialize ECE_Config_All_Data, which is a dictionary that contains all information that will be written to the parameter .json file.
        ECE_Config_All_Data = _orddict([
            ("User", self.AllNames['User'] ),
            ("Valid_Since", self.AllNames['Valid_Since'] ),
            ("Modified_At", str( utc2time( _time.time()*1e9 ) ) ),
            ("Frontend", self.AllNames['Frontend'] ),
            ("Band", self.AllNames['Band'] ),
            ("RotAttenuator_dB", self.AllNames['RotaryAttenuator_dB'] ),
            ("FixedAttenuator_dB", self.AllNames['FixedAttenuator_dB'] ),
            ("Sampling_Rate_KHz", self.AllNames['Fs'] ),            
            ("General_Remarks", self.AllNames['General_Remarks'] ),
            ("Channels", Channel_Data)
                    ])
        
        # ==== Dump ECE_Config_All_Data to a .json file with the filename as evaluated above.
        with open( sfilename, 'w') as fid:
            _jsn.dump(ECE_Config_All_Data, fid, sort_keys=False, indent=4, ensure_ascii=False)        
        #fileopen
             
        # ======  Print success message to the command line.
        if self.verbose:
            print('Successfully saved file '+sfilename) 
        # endif
    # end def
    
    # ====================================================================== #
      
    def getECEfreq(self):
        return self.RF, self.BW
    # end def getECEfreq

    # ====================================================================== #
        
# end class backend_config

# ========================================================================== #





