import os
import logging
import numpy as np
import pandas as pd
import configparser

RC_INTERVAL_PREFIX = 'INTERVAL'
RC_INTERVAL_IDLEN = 3
RC_INTERVAL_REPLEN = 4
RC_INTSUFFIX_POS = 'POS'
RC_INTSUFFIX_NEG = 'NEG'

def BuildSweep(minval, maxval, ppd, sort_type):
    numvals = int(np.ceil(ppd * np.log10(maxval / minval)) + 1)
    unique_vals = []
    for i in range(numvals):
        unique_vals.append(minval * np.power(10, i * 1.0 / ppd))
    res = []
    if (sort_type == 'ASC' or sort_type == 'ASCDESC'):
        res += unique_vals
    if (sort_type == 'DESC' or  sort_type == 'ASCDESC' or sort_type == 'DESCASC'):
        res += unique_vals[::-1]
    if (sort_type == 'DESCASC'):
        res += unique_vals
    return res


class RheoProtocol():

    def __init__(self, config_folder, config_fname, explog_fname=None, config_comments=';'):
        self.intervals = []
        self.upacked_intervals = None
        self.config_folder = config_folder
        self.config_fname = config_fname
        self.explog_fname = explog_fname
        self.LoadFromConfigFile(os.path.join(config_folder, config_fname), comment_prefix=config_comments)
        self.explogdata = None
        if explog_fname is not None:
            self.LoadExpLog()

    def __repr__(self):
        return '<RheoProtocol (' + str(len(self.intervals)) + ' intervals)>'
    
    def __str__(self):
        return '<RheoProtocol (' + str(len(self.intervals)) + ' intervals)>'
    
    def ToString(self, unpacked=False):
        str_res  = '\n|--------------------|'
        str_res += '\n| RheoProtocol class |'
        str_res += '\n|--------------------|'
        str_res += '\n| > Folder    : ' + str(self.config_folder)
        str_res += '\n| > AxisID    : ' + str(self.axID)
        str_res += '\n| > Gap       : ' + str(self.gap)
        str_res += '\n| > OutFolder : ' + str(self.outfolder)        
        str_res += '\n| > ' + str(len(self.intervals)) + ' Intervals'
        if unpacked:
            int_list = self.UnpackedIntervals()
        else:
            int_list = self.intervals
        if unpacked:
            str_res += ' (' + str(len(int_list)) + ' sub-intervals):'
        for i in range(len(int_list)):
            str_res += '\n|   [' + str(i).zfill(3) + '] ' + int_list[i].ToString()
        return str_res

    def UnpackedIntervals(self):
        if self.upacked_intervals is None:
            self.upacked_intervals = []
            for cur_int in self.intervals:
                if cur_int.active:
                    self.upacked_intervals += cur_int.unpackIntervals(self.fext)
        return self.upacked_intervals

    def CountIntervals(self, unpacked=True, check_explog=False):
        if unpacked:
            res = len(self.UnpackedIntervals())
            if check_explog and self.explogdata is not None:
                logrows = self.explogdata.shape[0]
                if logrows != res:
                    logging.warning('RheoConfig has ' + str(res) + ' unpacked intervals generated from config file ' + str(self.config_fname) +\
                                 ' and ' + str(logrows) + ' rows in expLog file ' + str(self.explog_fname))
                    return min(res, logrows)
            return res
        else:
            return len(self.intervals)

    def GetFullFilenames(self, use_froot=None, hide_disabled=True):
        if use_froot is None:
            use_froot = self.config_folder
        res = []
        for i, cur_int in enumerate(self.UnpackedIntervals()):
            if cur_int.filename is None:
                logging.error('Unpacked RheoInterval #' + str(i) + ' has no filename associated (namebase: ' + str(cur_int.namebase) + ')')
            else:
                if cur_int.active or hide_disabled==False:
                    res.append(os.path.join(use_froot, cur_int.filename))
        return res
    
    def LoadExpLog(self, fname=None, fheadersep='------------------------', force_reload=False):
        if force_reload or self.explogdata is None:
            if fname is None:
                fname = os.path.join(self.config_folder, self.explog_fname)
            if os.path.isfile(fname):
                res = []
                row_count = 0
                file_header = None
                with open(fname, 'r') as f:
                    if fheadersep is not None:
                        for line in f:
                            if fheadersep in line:
                                # now the interesting part begins. Skip the header and start reading
                                for line in f: 
                                    if row_count == 0:
                                        file_header = line.strip().split('\t')
                                    else:
                                        res.append(line.strip().split('\t'))
                                    row_count += 1
                    else: 
                        for line in f: 
                            if row_count == 0:
                                file_header = line.strip().split('\t')
                            else:
                                res.append(line.strip().split('\t'))
                            row_count += 1
                logging.info(str(row_count) + ' rows read from expLog file ' + str(fname))
                self.explogdata = pd.DataFrame(data=res, columns=file_header)
            else:
                logging.error('No expLog file found at path ' + str(fname))
        return self.explogdata

    def LoadFromConfigFile(self, fname, fext='.txt', comment_prefix=";"):
        """Reads the configuration file for rheology

        Parameters
        ----------
        - fname : full path of the *.ini configuration file used by rheology program
        - fext :  extension of output files

        Returns
        -------
        list of RheoInterval steps
        """
        
        self.intervals = []
        self.fext = fext
        
        config = configparser.ConfigParser(inline_comment_prefixes=comment_prefix)
        config.read(fname)

        self.gap = config.getfloat('EXPERIMENT', 'Gap', fallback=1.0)
        self.axID = config.getint('EXPERIMENT', 'AxisID', fallback=0)
        self.outfolder = config.get('EXPERIMENT', 'OutFolder', fallback='')
        
        for i in range(1, len(config.sections())):
            strsec = RC_INTERVAL_PREFIX + str(i)
            if strsec in config.sections():
                cur_name = str(i).zfill(2) + '_' + config[strsec]['Name']
                cur_active = config.getboolean(strsec, 'Active', fallback=1)
                cur_ax = config.getint(strsec, 'AxisID', fallback=self.axID)
                cur_type = config.get(strsec, 'Type', fallback='')
                cur_endaction = config.get(strsec, 'EndAction', fallback='NONE')
                cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                cur_osrtype = config.get(strsec, 'OSR_Type', fallback='')
                if cur_osrtype == 'OSCILL_POS':
                    cur_osramp = config.getfloat(strsec, 'OSR_Amplitude', fallback=0.0)
                    cur_osrperiod = config.getfloat(strsec, 'OSR_Period', fallback=0.0)
                    cur_osr = {'Type' : cur_osrtype, 'Amp' : cur_osramp, 'Period' : cur_osrperiod}
                else:
                    cur_osr = None
                if cur_active:
                    if cur_type == 'OSCILL_POS':
                        cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                        cur_period = config.getfloat(strsec, 'Period', fallback=-1.0)
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, Active=cur_active, OSR=cur_osr, filename=cur_name, amplitude=cur_amp, 
                                                           period=cur_period, offset=cur_offset, reptimes=cur_reptimes))
                    elif cur_type == 'SWEEP_FREQ':
                        cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                        cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                        min_freq = config.getfloat(strsec, 'MinFreq', fallback=-1.0)
                        max_freq = config.getfloat(strsec, 'MaxFreq', fallback=-1.0)
                        cur_sort = config.get(strsec, 'FreqSort', fallback='ASC')
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, Active=cur_active, OSR=cur_osr, namebase=cur_name, amplitude=cur_amp, 
                                                            minfreq=min_freq, maxfreq=max_freq, ppd=cur_ppd, sort=cur_sort, 
                                                            offset=cur_offset, reptimes=cur_reptimes))

                    elif cur_type == 'SWEEP_STRAIN':
                        cur_freq = config.getfloat(strsec, 'AngularFrequency', fallback=-1.0)
                        cur_period = 2*np.pi/cur_freq
                        cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                        min_amp = config.getfloat(strsec, 'MinAmplitude', fallback=-1.0)
                        max_amp = config.getfloat(strsec, 'MaxAmplitude', fallback=-1.0)
                        cur_sort = config.get(strsec, 'AmpSort', fallback='ASC')
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, Active=cur_active, OSR=cur_osr, namebase=cur_name, freq=cur_freq, 
                                                            minamp=min_amp, maxamp=max_amp, ppd=cur_ppd, sort=cur_sort, 
                                                            offset=cur_offset, reptimes=cur_reptimes))

                    elif cur_type == 'STEP_RATE':
                        cur_rate = config.getfloat(strsec, 'ShearRate', fallback=-1.0)
                        cur_totstrain = config.getfloat(strsec, 'TotalStrain', fallback=-1.0)
                        cur_revafter = config.getboolean(strsec, 'ReverseAfter', fallback=1)
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, Active=cur_active, OSR=cur_osr, namebase=cur_name, rate=cur_rate, 
                                                           strain=cur_totstrain, twoways=cur_revafter, offset=cur_offset, reptimes=cur_reptimes))

                    elif cur_type == 'SWEEP_RATE':
                        cur_minrate = config.getfloat(strsec, 'MinShearRate', fallback=-1.0)
                        cur_maxrate = config.getfloat(strsec, 'MaxShearRate', fallback=-1.0)
                        cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                        cur_sort = config.get(strsec, 'RateSort', fallback='ASC')
                        cur_runsperrate = int(config.getfloat(strsec, 'RunsPerShearRate', fallback=0.0))
                        cur_totstrain = config.getfloat(strsec, 'TotalStrain', fallback=-1.0)
                        cur_revafter = config.getboolean(strsec, 'ReverseAfter', fallback=1)
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, Active=cur_active, OSR=cur_osr, namebase=cur_name, minrate=cur_minrate, 
                                                           maxrate=cur_maxrate, ppd=cur_ppd, sort=cur_sort, strain=cur_totstrain, runsperrate=cur_runsperrate, 
                                                           twoways=cur_revafter, offset=cur_offset, reptimes=cur_reptimes))
                    elif cur_type == 'STEP_POS':
                        cur_rate = config.getfloat(strsec, 'ShearRate', fallback=-1.0)
                        cur_amp = config.getfloat(strsec, 'Amplitude', fallback=0.0)
                        cur_dur = config.getfloat(strsec, 'Duration', fallback=0.0)
                        cur_stept = config.getfloat(strsec, 'StepTime', fallback=0.0)
                        cur_rect = config.getfloat(strsec, 'RecoveryTime', fallback=0.0)
                        cur_revafter = config.getboolean(strsec, 'ReverseAfter', fallback=1)
                        self.intervals.append(RheoInterval(cur_type, AxID=cur_ax, OSR=cur_osr, namebase=cur_name, amplitude=cur_amp, 
                                                           duration=cur_dur, twoways=cur_revafter, offset=cur_offset, reptimes=cur_reptimes))
                        
                    else:
                        logging.error('unknown measure type ' + str(config[strsec]['Type']) + ' in section ' + str(strsec))

class RheoInterval():
    """ Bundles rheo interval properties """

    def __init__(self, IntervalType=None, AxID=0, Active=1, OSR=None, **kwdict):
        self.type = IntervalType
        self.axID = AxID
        self.active = Active
        self.OSR = OSR
        if (IntervalType == 'OSCILL_POS'):
            self._init_oscillPos(**kwdict)
        elif (IntervalType == 'SWEEP_FREQ'):
            self._init_DFS(**kwdict)
        elif (IntervalType == 'SWEEP_STRAIN'):
            self._init_DSS(**kwdict)
        elif (IntervalType == 'STEP_RATE'):
            self._init_stepRate(**kwdict)
        elif (IntervalType == 'SWEEP_RATE'):
            self._init_rateSweep(**kwdict)
        elif (IntervalType == 'STEP_POS'):
            self._init_stepPos(**kwdict)
        else:
            raise ValueError('RheoInterval type ' + str(self.type) + ' not recognized')

    
    def _init_oscillPos(self, amplitude, period, offset, reptimes, namebase=None, filename=None):
        self.filename = filename
        if filename is not None:
            self.namebase = filename
            self.name = filename
        else:
            self.namebase = namebase
            self.name = namebase
        self.amplitude = amplitude
        self.period = period
        self.offset = offset
        self.reptimes = reptimes

    def _init_DFS(self, namebase, amplitude, minfreq, maxfreq, ppd, sort, offset, reptimes):
        self.namebase = namebase
        self.amplitude = amplitude
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.ppd = ppd
        self.numfreq = int(np.ceil(ppd * np.log10(maxfreq / minfreq)) + 1)
        self.sort = sort
        self.offset = offset
        self.reptimes = reptimes
        self.name = self.namebase

    def _init_DSS(self, namebase, freq, minamp, maxamp, ppd, sort, offset, reptimes):
        self.namebase = namebase
        self.freq = freq
        self.minamp = minamp
        self.maxamp = maxamp
        self.ppd = ppd
        self.numamps = int(np.ceil(ppd * np.log10(maxamp / minamp)) + 1)
        self.sort = sort
        self.offset = offset
        self.reptimes = reptimes
        self.name = self.namebase

    def _init_stepRate(self, rate, strain, twoways, offset, reptimes, namebase=None, filename=None):
        self.filename = filename
        if filename is not None:
            self.namebase = filename
            self.name = filename
        else:
            self.namebase = namebase
            self.name = namebase
        self.rate = rate
        self.strain = strain
        self.offset = offset
        self.twoways = twoways
        self.reptimes = reptimes

    def _init_rateSweep(self, namebase, minrate, maxrate, ppd, sort, strain, runsperrate, twoways, offset, reptimes):
        self.namebase = namebase
        self.minrate = minrate
        self.maxrate = maxrate
        self.ppd = ppd
        self.numrates = int(np.ceil(ppd * np.log10(maxrate / minrate)) + 1)
        self.sort = sort
        self.runsperrate = runsperrate
        self.strain = strain
        self.offset = offset
        self.twoways = twoways
        self.reptimes = reptimes
        self.name = self.namebase
        
    def _init_stepPos(self, amplitude, duration, twoways, offset, reptimes, namebase=None, filename=None):
        self.filename = filename
        if filename is not None:
            self.namebase = filename
            self.name = filename
        else:
            self.namebase = namebase
            self.name = namebase
        self.amplitude = amplitude
        self.duration = duration
        self.offset = offset
        self.reptimes = reptimes
        self.twoways = twoways

    def Copy(self):
        if (self.type == 'OSCILL_POS'):
            return RheoInterval(self.type, self.active, namebase=self.namebase, filename=self.filename, amplitude=self.amplitude, 
                                period=self.period, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_FREQ'):
            return RheoInterval(self.type, self.active, namebase=self.namebase, amplitude=self.amplitude, minfreq=self.minfreq, 
                                maxfreq=self.maxfreq, ppd=self.ppd, sort=self.sort, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_STRAIN'):
            return RheoInterval(self.type, self.active, namebase=self.namebase, freq=self.freq, minamp=self.minamp, 
                                maxamp=self.maxamp, ppd=self.ppd, sort=self.sort, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'STEP_RATE'):
            return RheoInterval(self.type, self.active, namebase=self.namebase, filename=self.filename, rate=self.rate, strain=self.strain, 
                                twoways=self.twoways, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_RATE'):
            return RheoInterval(self.type, self.active, namebase=self.namebase, minrate=self.minrate, maxrate=self.maxrate, ppd=self.ppd, 
                                sort=self.sort, strain=self.strain, runsperrate=self.runsperrate, offset=self.offset, reptimes=self.reptimes)
        else:
            raise ValueError('Unable to copy interval of type ' + str(self.type))

    def __repr__(self):
        str_res = '<RheoInterval: ' + str(self.type) + ' (' + self.name + ')'
        return str_res + '>'
    
    def __str__(self):
        return '<RheoInterval: ' + self.ToString() + '>'
    
    def ToString(self):
        str_res = str(self.type) + ' (' + self.name + ') - '
        if (self.type == 'OSCILL_POS'):
            str_res += 'A=' + str(self.amplitude) + '; T=' + str(self.period)
        elif (self.type == 'SWEEP_FREQ'):
            str_res += 'A=' + str(self.amplitude) + '; w=[' + str(self.minfreq) + ',' + str(self.maxfreq) + ']; ' + str(self.numfreq) + ' pts ' + str(self.sort)
        elif (self.type == 'SWEEP_STRAIN'):
            str_res += 'w=' + str(self.freq) + '; A=[' + str(self.minamp) + ',' + str(self.maxamp) + ']; ' + str(self.numamps) + ' pts ' + str(self.sort)
        elif (self.type == 'STEP_RATE'):
            str_res += 'v=' + str(self.rate) + '; totx=' + str(self.strain)
            if self.twoways:
                str_res += ' (a/r)'
        elif (self.type == 'SWEEP_RATE'):
            str_res += 'v=[' + str(self.minrate) + ',' + str(self.maxrate) + ']; ' + str(self.numrates) + ' pts ' + str(self.sort) + '; ' + str(self.runsperrate) + ' runs'
        str_res += '; off=' + str(self.offset) + '; rep ' + str(self.reptimes) + 'x'
        if not self.active:
            str_res += '; DISABLED'
        return str_res

    def unpackIntervals(self, fext='.txt'):
        res = []
        if (self.type == 'OSCILL_POS'):
            for j in range(self.reptimes):
                if self.reptimes > 1:
                    cur_name = self.namebase + '_' + str(j).zfill(RC_INTERVAL_REPLEN) + fext
                else:
                    cur_name = self.namebase + fext
                res.append(RheoInterval(self.type, self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_name, 
                                                amplitude=self.amplitude, period=self.period, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_FREQ'):
            all_freq = BuildSweep(self.minfreq, self.maxfreq, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_freq)):
                    cur_freq = all_freq[i]
                    cur_period = 2*np.pi/cur_freq
                    res.append(RheoInterval('OSCILL_POS', self.axID, self.active, self.OSR, namebase=self.namebase, filename=self.namebase + '_' + str(i).zfill(RC_INTERVAL_IDLEN) + chr(j+97) + fext, amplitude=self.amplitude, period=cur_period, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_STRAIN'):
            cur_period = 2*np.pi/self.freq
            all_amp = BuildSweep(self.minamp, self.maxamp, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_amp)):
                    cur_amp = all_amp[i]
                    res.append(RheoInterval('OSCILL_POS', self.axID, self.active, self.OSR, namebase=self.namebase, filename=self.namebase + '_' + str(i).zfill(RC_INTERVAL_IDLEN) + chr(j+97) + fext, amplitude=cur_amp, period=cur_period, offset=self.offset, reptimes=1))
        elif (self.type == 'STEP_RATE'):
            for j in range(self.reptimes):
                if (self.reptimes > 1):
                    cur_fname = self.namebase + '_' + str(j).zfill(RC_INTERVAL_IDLEN)
                else:
                    cur_fname = self.namebase
                cur_fname_asc = cur_fname
                if self.twoways:
                    cur_fname_asc += '_'+RC_INTSUFFIX_POS
                res.append(RheoInterval(self.type, self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname_asc+fext, rate=self.rate, strain=self.strain, 
                                        twoways=False, offset=self.offset, reptimes=1))
                if self.twoways:
                    res.append(RheoInterval(self.type, self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname+'_'+RC_INTSUFFIX_NEG+fext, rate=self.rate, 
                                            strain=self.strain, twoways=False, offset=self.offset, reptimes=1))
        elif (self.type == 'STEP_POS'):
            for j in range(self.reptimes):
                if (self.reptimes > 1):
                    cur_fname = self.namebase + '_' + str(j).zfill(RC_INTERVAL_IDLEN)
                else:
                    cur_fname = self.namebase
                cur_fname_asc = cur_fname
                if self.twoways:
                    cur_fname_asc += '_'+RC_INTSUFFIX_POS
                res.append(RheoInterval(self.type, self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname_asc+fext, amplitude=self.amplitude, duration=self.duration, 
                                        twoways=False, offset=self.offset, reptimes=1))
                if self.twoways:
                    res.append(RheoInterval(self.type, self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname+'_'+RC_INTSUFFIX_NEG+fext, amplitude=self.amplitude, 
                                            duration=self.duration, twoways=False, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_RATE'):
            all_rates = BuildSweep(self.minrate, self.maxrate, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_rates)):
                    for k in range(self.runsperrate):
                        cur_fname = self.namebase + '_' + str(i).zfill(RC_INTERVAL_IDLEN) + chr(j+97)
                        if self.runsperrate > 1:
                             cur_fname += '_' + str(k).zfill(RC_INTERVAL_REPLEN)
                        if self.twoways:
                            res.append(RheoInterval('STEP_RATE', self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname+'_'+RC_INTSUFFIX_POS+fext, rate=all_rates[i], strain=self.strain, 
                                                    twoways=False, offset=self.offset, reptimes=1))
                            res.append(RheoInterval('STEP_RATE', self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname+'_'+RC_INTSUFFIX_NEG+fext, rate=all_rates[i], 
                                                    strain=self.strain, twoways=False, offset=self.offset, reptimes=1))
                        else:
                            res.append(RheoInterval('STEP_RATE', self.axID, self.active, self.OSR, namebase=self.namebase, filename=cur_fname+fext, rate=all_rates[i], strain=self.strain, 
                                                    twoways=False, offset=self.offset, reptimes=1))
        else:
            raise ValueError('RheoInterval type ' + str(self.type) + ' not recognized')
        return res