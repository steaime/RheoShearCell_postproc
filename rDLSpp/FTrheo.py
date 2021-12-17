import copy
import configparser
import numpy as np
from rDLSpp import IOfuncs as iof

"""
Parameters:
----------
- fname:          string, full path of the file to be analyzed
- Period:         float, oscillation period, in seconds
- StartTime:      float, where to start the analysis, in units of the period.
- AnalyzePeriods: int, number of periods to analyze
- FreqRecord:     int, how many datapoints per second. If none, it will be calculated automatically
- ForceCorrection:function or None
"""
def FTanalysisRheology(fname, Period=1.0, StartTime=1.0, AnalyzePeriods=1, FreqRecord=None, StaticForceCorr=None, CurrentForceCorr=None, verbose=0, savefile_suf=None):
    # Period: Oscillation period, in seconds
    # FreqRecord: How many points per second
    ManualTimeDelay_sec = 0.0 # Eventually, consider a time delay between force and position readings

    # Load information on Rheodiff profiles
    # Also load raw data if different resampled profiles were saved on the same output file.
    t_list, x_list, f_list = iof.ReadRheoData(fname)
    if savefile_suf is not None:
        f_list_raw = copy.deepcopy(f_list)
    if StaticForceCorr is not None:
        f_static = StaticForceCorr(x_list)
        f_list -= f_static
    elif savefile_suf is not None:
        f_static = np.zeros_like(f_list)
    if CurrentForceCorr is not None:
        f_currFactor = CurrentForceCorr(x_list)
        f_list /= f_currFactor
    elif savefile_suf is not None:
        f_currFactor = np.ones_like(f_list)
    
    if FreqRecord is None:
        FreqRecord = 1000.0 / (t_list[1] - t_list[0])
    if Period is None:
        raise ValueError('Period cannot be None')
      
    # STEP 1: Load each dataset and select subsets equal to N oscillation periods
    # For each subset, calculate FFT and store it (amplitude and phase)
    # also, save time-averaged data in one file.
    # Note: in order to work properly with FFT algorithm,window must be defined with one open boundary: [t, t+NT)                    
    PointsPerPeriod = int(FreqRecord * Period)
    WindowEdges = (StartTime, StartTime+AnalyzePeriods) # [t_min, t_max], in units of the period
    IndexEdges = (int(WindowEdges[0] * PointsPerPeriod), int(WindowEdges[1] * PointsPerPeriod))
    NumPeriods = int(WindowEdges[1] - WindowEdges[0])
    PointsPerWindow = PointsPerPeriod * NumPeriods
    
    if verbose>0:
        print('FT analysis on {0} datapoints (period: {1}, record frequency: {2}, {3} periods, {4} points per window)'.format(len(t_list), Period, FreqRecord, NumPeriods, PointsPerWindow))
    
    # to get real amplitudes from FFT you have to divide by half the channel number
    PositionFFT = np.fft.rfft(x_list[IndexEdges[0]:IndexEdges[1]]) / PointsPerWindow * 2
    ForceFFT = np.fft.rfft(f_list[IndexEdges[0]:IndexEdges[1]]) / PointsPerWindow * 2
    AvgForce = np.mean(f_list[IndexEdges[0]:IndexEdges[1]])
    
    # process first harmonic to get moduli
    PositionFFT_FirstHarm = PositionFFT[NumPeriods]
    ForceFFT_FirstHarm = ForceFFT[NumPeriods]
    G_FirstHarm = ForceFFT_FirstHarm/PositionFFT_FirstHarm
    
    # Eventually correct G for time delay btw force and position readings
    PhaseDiff = 2*np.pi*ManualTimeDelay_sec*1.0/Period
    ComplexTransform = np.cos(PhaseDiff) + 1j*np.sin(PhaseDiff)
    G_FirstHarm *= ComplexTransform

    if verbose>1:
        print('  F_amp   = ' + str(np.abs(ForceFFT_FirstHarm)) + ' N')
        print('  F_phase = ' + str(np.angle(ForceFFT_FirstHarm)) + ' rad')
        print('  x_amp   = ' + str(np.abs(PositionFFT_FirstHarm)) + ' mm')
        print('  x_phase = ' + str(np.angle(PositionFFT_FirstHarm)) + ' rad')
        print('  k_amp   = ' + str(np.abs(G_FirstHarm)) + ' N/mm')
        print('  k_phase = ' + str(np.angle(G_FirstHarm)) + ' rad')
        print('  k_real  = ' + str(np.real(G_FirstHarm)) + ' N/mm')
        print('  k_imag  = ' + str(np.imag(G_FirstHarm)) + ' N/mm')
        print('  tandelta= ' + str(np.tan(np.angle(G_FirstHarm))) + ' N/mm')
        
    if savefile_suf is not None:
        np.savetxt(fname[:-4] + savefile_suf + fname[-4:], np.column_stack((t_list-t_list[0], x_list, f_list_raw, f_static, f_currFactor, f_list)), delimiter='\t', header='t[ms]\tpos[mm]\tFraw[N]\tFstatic[N]\tFIconversion\tFcorr[N]')
    
    return ForceFFT_FirstHarm, PositionFFT_FirstHarm, G_FirstHarm, AvgForce


'''
Returns: list of file specs, in the form [res_filename, amplitude, period, offset]
'''
def ReadConfig(fname, fext='.txt'):
    res = []
    
    config = configparser.ConfigParser(inline_comment_prefixes=";")
    config.read(fname)
    
    for i in range(1, len(config.sections())):
        strsec = 'MEASURE' + str(i)
        if strsec in config.sections():
            cur_type = config.get(strsec, 'Type', fallback='')
            if cur_type == 'OSCILL_POS':
                cur_name = str(i).zfill(2) + '_' + config[strsec]['Name'] + '.txt'
                cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                cur_period = config.getfloat(strsec, 'Period', fallback=-1.0)
                cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                res.append([cur_name, cur_amp, cur_period, cur_offset])
            elif cur_type == 'SWEEP_FREQ':
                cur_namebase = str(i).zfill(2) + '_' + config[strsec]['Name']
                cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                cur_amp = config.getfloat(strsec, 'Amplitude', fallback=-1.0)
                cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                min_freq = config.getfloat(strsec, 'MinFreq', fallback=-1.0)
                max_freq = config.getfloat(strsec, 'MaxFreq', fallback=-1.0)
                num_freq = int(np.ceil(cur_ppd * np.log10(max_freq / min_freq)) + 1)
                cur_sort = config.get(strsec, 'FreqSort', fallback='ASC')
                if cur_sort=='ASC':
                    inc_sign = 1.0
                    start_freq = min_freq
                else:
                    inc_sign = -1.0
                    start_freq = max_freq
                for j in range(cur_reptimes):
                    for i in range(num_freq):
                        cur_freq = start_freq * np.power(10, i * inc_sign / cur_ppd)
                        cur_period = 2*np.pi/cur_freq
                        res.append([cur_namebase + '_' + str(i).zfill(3) + chr(j+97) + fext, cur_amp, cur_period, cur_offset])
            elif cur_type == 'SWEEP_STRAIN':
                cur_namebase = str(i).zfill(2) + '_' + config[strsec]['Name']
                cur_reptimes = config.getint(strsec, 'RepeatTimes', fallback=1)
                cur_freq = config.getfloat(strsec, 'AngularFrequency', fallback=-1.0)
                cur_period = 2*np.pi/cur_freq
                cur_offset = config.getfloat(strsec, 'Offset', fallback=-1.0)
                cur_ppd = int(config.getfloat(strsec, 'PointsPerDecade', fallback=0.0))
                min_amp = config.getfloat(strsec, 'MinAmplitude', fallback=-1.0)
                max_amp = config.getfloat(strsec, 'MaxAmplitude', fallback=-1.0)
                num_amps = int(np.ceil(cur_ppd * np.log10(max_amp / min_amp)) + 1)
                cur_sort = config.get(strsec, 'AmpSort', fallback='ASC')
                if cur_sort=='ASC':
                    inc_sign = 1.0
                    start_amp = min_amp
                else:
                    inc_sign = -1.0
                    start_amp = max_amp
                for j in range(cur_reptimes):
                    for i in range(num_amps):
                        cur_amp = start_amp * np.power(10, i * inc_sign / cur_ppd)
                        res.append([cur_namebase + '_' + str(i).zfill(3) + chr(j+97) + fext, cur_amp, cur_period, cur_offset])
            else:
                print('unknown measure type ' + str(config[strsec]['Type']) + ' in section ' + str(strsec))
    
    return res