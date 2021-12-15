import logging
import numpy as np
import configparser

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

    def __init__(self, configfile, config_comments=';'):
        self.intervals = []
        self.LoadFromConfigFile(configfile, comment_prefix=config_comments)

    def __repr__(self):
        return '<RheoProtocol: (' + str(len(self.intervals)) + ' intervals)>'
    
    def __str__(self):
        return '<RheoProtocol: (' + str(len(self.intervals)) + ' intervals)>'
    
    def ToString(self, unpacked=False):
        if unpacked:
            int_list = self.UnpackIntervals()
        else:
            int_list = self.intervals
        str_res = str(len(self.intervals)) + ' intervals'
        if unpacked:
            str_res += '; ' + str(len(int_list)) + ' sub-intervals:'
        for cur_int in int_list:
            str_res += '\n- ' + cur_int.ToString()
        return str_res

    def UnpackIntervals(self):
        res = []
        for cur_int in self.intervals:
            res += cur_int.unpackIntervals(self.fext)
        return res

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
        
        for i in range(1, len(config.sections())):
            strsec = 'MEASURE' + str(i)
            if strsec in config.sections():
                cur_type = config.get(strsec, 'Type', fallback='')
                if cur_type == 'OSCILL_POS':
                    cur_name = str(i).zfill(2) + '_' + config[strsec]['Name']
                    cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                    cur_period = config.getfloat(strsec, 'Period', fallback=-1.0)
                    cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                    cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                    self.intervals.append(RheoInterval(cur_type, filename=cur_name, amplitude=cur_amp, period=cur_period, 
                                                offset=cur_offset, reptimes=cur_reptimes))
                elif cur_type == 'SWEEP_FREQ':
                    cur_namebase = str(i).zfill(2) + '_' + config[strsec]['Name']
                    cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                    cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                    cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                    cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                    min_freq = config.getfloat(strsec, 'MinFreq', fallback=-1.0)
                    max_freq = config.getfloat(strsec, 'MaxFreq', fallback=-1.0)
                    cur_sort = config.get(strsec, 'FreqSort', fallback='ASC')
                    self.intervals.append(RheoInterval(cur_type, namebase=cur_namebase, amplitude=cur_amp, 
                                                        minfreq=min_freq, maxfreq=max_freq, ppd=cur_ppd, sort=cur_sort, 
                                                        offset=cur_offset, reptimes=cur_reptimes))

                elif cur_type == 'SWEEP_STRAIN':
                    cur_namebase = str(i).zfill(2) + '_' + config[strsec]['Name']
                    cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                    cur_freq = config.getfloat(strsec, 'AngularFrequency', fallback=-1.0)
                    cur_period = 2*np.pi/cur_freq
                    cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                    cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                    min_amp = config.getfloat(strsec, 'MinAmplitude', fallback=-1.0)
                    max_amp = config.getfloat(strsec, 'MaxAmplitude', fallback=-1.0)
                    cur_sort = config.get(strsec, 'AmpSort', fallback='ASC')
                    self.intervals.append(RheoInterval(cur_type, namebase=cur_namebase, freq=cur_freq, 
                                                        minamp=min_amp, maxamp=max_amp, ppd=cur_ppd, sort=cur_sort, 
                                                        offset=cur_offset, reptimes=cur_reptimes))

                elif cur_type == 'STEP_RATE':
                    cur_name = str(i).zfill(2) + '_' + config[strsec]['Name']
                    cur_rate = config.getfloat(strsec, 'ShearRate', fallback=-1.0)
                    cur_totstrain = config.getfloat(strsec, 'TotalStrain', fallback=-1.0)
                    cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                    cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                    cur_revafter = config.getboolean(strsec, 'ReverseAfter', fallback=1)
                    self.intervals.append(RheoInterval(cur_type, namebase=cur_name, rate=cur_rate, strain=cur_totstrain, 
                                                twoways=cur_revafter, offset=cur_offset, reptimes=cur_reptimes))

                elif cur_type == 'SWEEP_RATE':
                    cur_name = str(i).zfill(2) + '_' + config[strsec]['Name']
                    cur_minrate = config.getfloat(strsec, 'MinShearRate', fallback=-1.0)
                    cur_maxrate = config.getfloat(strsec, 'MaxShearRate', fallback=-1.0)
                    cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                    cur_sort = config.get(strsec, 'RateSort', fallback='ASC')
                    cur_runsperrate = int(config.getfloat(strsec, 'RunsPerShearRate', fallback=0.0))
                    cur_totstrain = config.getfloat(strsec, 'TotalStrain', fallback=-1.0)
                    cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                    cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                    self.intervals.append(RheoInterval(cur_type, namebase=cur_name, minrate=cur_minrate, maxrate=cur_maxrate, ppd=cur_ppd, 
                                                        sort=cur_sort, strain=cur_totstrain, runsperrate=cur_runsperrate, offset=cur_offset, reptimes=cur_reptimes))
                else:
                    logging.error('unknown measure type ' + str(config[strsec]['Type']) + ' in section ' + str(strsec))

class RheoInterval():
    """ Bundles rheo interval properties """

    def __init__(self, IntervalType=None, **kwdict):
        self.type = IntervalType
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
        else:
            raise ValueError('RheoInterval type ' + str(self.type) + ' not recognized')

    
    def _init_oscillPos(self, filename, amplitude, period, offset, reptimes):
        self.filename = filename
        self.namebase = filename
        self.amplitude = amplitude
        self.period = period
        self.offset = offset
        self.reptimes = reptimes
        self.name = self.filename

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

    def _init_stepRate(self, namebase, rate, strain, twoways, offset, reptimes):
        self.namebase = namebase
        self.rate = rate
        self.strain = strain
        self.offset = offset
        self.twoways = twoways
        self.reptimes = reptimes
        self.name = self.namebase

    def _init_rateSweep(self, namebase, minrate, maxrate, ppd, sort, strain, runsperrate, offset, reptimes):
        self.namebase = namebase
        self.minrate = minrate
        self.maxrate = maxrate
        self.ppd = ppd
        self.numrates = int(np.ceil(ppd * np.log10(maxrate / minrate)) + 1)
        self.sort = sort
        self.runsperrate = runsperrate
        self.strain = strain
        self.offset = offset
        self.reptimes = reptimes
        self.name = self.namebase


    def Copy(self):
        if (self.type == 'OSCILL_POS'):
            return RheoInterval(self.type, filename=self.filename, amplitude=self.amplitude, 
                                period=self.period, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_FREQ'):
            return RheoInterval(self.type, namebase=self.namebase, amplitude=self.amplitude, minfreq=self.minfreq, 
                                maxfreq=self.maxfreq, ppd=self.ppd, sort=self.sort, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_STRAIN'):
            return RheoInterval(self.type, namebase=self.namebase, freq=self.freq, minamp=self.minamp, 
                                maxamp=self.maxamp, ppd=self.ppd, sort=self.sort, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'STEP_RATE'):
            return RheoInterval(self.type, namebase=self.filename, rate=self.rate, strain=self.strain, 
                                twoways=self.twoways, offset=self.offset, reptimes=self.reptimes)
        elif (self.type == 'SWEEP_RATE'):
            return RheoInterval(self.type, namebase=self.filename, minrate=self.minrate, maxrate=self.maxrate, ppd=self.ppd, 
                                sort=self.sort, strain=self.strain, runsperrate=self.runsperrate, offset=self.offset, reptimes=self.reptimes)
        else:
            raise ValueError('Unable to copy interval of type ' + str(self.type))

    def __repr__(self):
        return '<RheoInterval: ' + str(self.type) + ' (' + self.name + ')>'
    
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
        return str_res

    def unpackIntervals(self, fext='.txt'):
        res = []
        if (self.type == 'OSCILL_POS'):
            for j in range(self.reptimes):
                res.append(RheoInterval(self.type, filename=self.namebase + '_' + str(j).zfill(4) + fext, 
                                                amplitude=self.amplitude, period=self.period, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_FREQ'):
            all_freq = BuildSweep(self.minfreq, self.maxfreq, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_freq)):
                    cur_freq = all_freq[i]
                    cur_period = 2*np.pi/cur_freq
                    res.append(RheoInterval('OSCILL_POS', filename=self.namebase + '_' + str(i).zfill(3) + chr(j+97) + fext, 
                                                amplitude=self.amplitude, period=cur_period, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_STRAIN'):
            cur_period = 2*np.pi/self.freq
            all_amp = BuildSweep(self.minamp, self.maxamp, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_amp)):
                    cur_amp = all_amp[i]
                    res.append(RheoInterval('OSCILL_POS', filename=self.namebase + '_' + str(i).zfill(3) + chr(j+97) + fext, 
                                                amplitude=cur_amp, period=cur_period, offset=self.offset, reptimes=1))
        elif (self.type == 'STEP_RATE'):
            for j in range(self.reptimes):
                cur_fname = self.namebase + '_' + str(j).zfill(4)
                cur_fname_asc = cur_fname
                if self.twoways:
                    cur_fname_asc += '_ASC'
                res.append(RheoInterval(self.type, namebase=cur_fname_asc+fext, rate=self.rate, strain=self.strain, 
                                        twoways=False, offset=self.offset, reptimes=1))
                if self.twoways:
                    res.append(RheoInterval(self.type, namebase=cur_fname+'_DESC'+fext, rate=self.rate, 
                                            strain=self.strain, twoways=False, offset=self.offset, reptimes=1))
        elif (self.type == 'SWEEP_RATE'):
            all_rates = BuildSweep(self.minrate, self.maxrate, self.ppd, self.sort)
            for j in range(self.reptimes):
                for i in range(len(all_rates)):
                    for k in range(self.runsperrate):
                        cur_fname = self.namebase + '_' + str(i).zfill(3) + chr(j+97) + '_' + str(k).zfill(4)
                        res.append(RheoInterval('STEP_RATE', namebase=cur_fname+'_ASC'+fext, rate=all_rates[i], strain=self.strain, 
                                                twoways=False, offset=self.offset, reptimes=1))
                        res.append(RheoInterval('STEP_RATE', namebase=cur_fname+'_DESC'+fext, rate=all_rates[i], 
                                                strain=self.strain, twoways=False, offset=self.offset, reptimes=1))
        else:
            raise ValueError('RheoInterval type ' + str(self.type) + ' not recognized')
        return res