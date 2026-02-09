import copy
import configparser
import numpy as np
import bisect
import logging
from rDLSpp import IOfuncs as iof
from rDLSpp import RheoCorr as rhc

def CalcTimeDependentModuli(fname, Period=1.0, StartTime=0.25, StepTime=0.25, AnalyzePeriods=1, Duration=None, verbose=0, usecols=(1,2,6), **loadtxt_kwargs):
    if StepTime==0:
        return FTanalysisRheology(fname, Period=Period, StartTime=StartTime, AnalyzePeriods=AnalyzePeriods, verbose=verbose, usecols=usecols, **loadtxt_kwargs)
    else:
        if Duration is None:
            t_list, x_list, f_list = iof.ReadRheoData(fname, usecols, unpack=True, **loadtxt_kwargs)
            Duration = (t_list[-1]-t_list[0])/1000
        Dur_periods = Duration/Period
        npoints = int(np.floor((Dur_periods-AnalyzePeriods-StartTime)/StepTime))
        if npoints > 0:
            res = np.empty((npoints, 3), dtype=float)
            for i in range(npoints):
                curt_periods = StartTime+i*StepTime
                G, _ =  FTanalysisRheology(fname, Period=Period, StartTime=curt_periods, AnalyzePeriods=AnalyzePeriods, FreqRecord=None, ForceCorrection=None, HigherHarmonics=0, 
                                           return_spectrum=False, verbose=verbose, usecols=usecols, **loadtxt_kwargs)
                res[i] = (curt_periods*Period, np.real(G), np.imag(G))
            return res
        else:
            return None
    

"""
Parameters:
----------
- fname:          string, full path of the file to be analyzed
- Period:         float, oscillation period, in seconds
- StartTime:      float, where to start the analysis, in units of the period.
- AnalyzePeriods: int, number of periods to analyze
- FreqRecord:     int, how many datapoints per second. If none, it will be calculated automatically
- ForceCorrection:RheoCorr.ForceCorrection or None
"""
def FTanalysisRheology(fname, Period=1.0, StartTime=1.0, AnalyzePeriods=1, FreqRecord=None, ForceCorrection=None, HigherHarmonics=6, return_spectrum=False, verbose=0, usecols=(1,2,6), **loadtxt_kwargs):

    # Period: Oscillation period, in seconds
    # FreqRecord: How many points per second
    ManualTimeDelay_sec = 0.0 # Eventually, consider a time delay between force and position readings

    # Load information on Rheodiff profiles
    # Also load raw data if different resampled profiles were saved on the same output file.
    t_list, x_list, f_list = iof.ReadRheoData(fname, usecols, unpack=True, **loadtxt_kwargs)
    if ForceCorrection is not None:
        ForceCorrection.Correct(f_list, x_list, inplace=True)
    
    if FreqRecord is None:
        FreqRecord = 1000.0 / (t_list[1] - t_list[0])
    if Period is None:
        raise ValueError('Period cannot be None')
      
    # STEP 1: Load each dataset and select subsets equal to N oscillation periods
    # For each subset, calculate FFT and store it (amplitude and phase)
    # also, save time-averaged data in one file.
    # Note: in order to work properly with FFT algorithm,window must be defined with one open boundary: [t, t+NT)
    PointsPerPeriod = int(FreqRecord * Period)
    if PointsPerPeriod <= 0:
        raise ValueError('Error processing file {0}: Points per period={1}, FreqRecord={2}, Period={3}'.format(fname, PointsPerPeriod, FreqRecord, Period))
    WindowEdges = (StartTime, StartTime+AnalyzePeriods) # [t_min, t_max], in units of the period
    #IndexEdges = (int(WindowEdges[0] * PointsPerPeriod), int(WindowEdges[1] * PointsPerPeriod))
    IndexEdges = [bisect.bisect(t_list-t_list[0], winedge*Period*1e3) for winedge in WindowEdges]
    NumPeriods = int(WindowEdges[1] - WindowEdges[0])
    PointsPerWindow = IndexEdges[1] - IndexEdges[0]
    if NumPeriods <= 0 or PointsPerWindow <= 0 or PointsPerWindow < FreqRecord * Period * NumPeriods - 1:
        logging.warning('Error processing file {0} ({1} time points): PointsPerWindow={2} (IndexEdges={3}), NumPeriods={4} (start: {5}, len: {6}, WindowEdges={7}), PointsPerPeriod={8} (T={9}, {10} fps)'.format(fname, len(t_list), PointsPerWindow, IndexEdges, NumPeriods, StartTime, AnalyzePeriods, WindowEdges, PointsPerPeriod, Period, FreqRecord))
        return np.nan, {'F': np.nan, 'x': np.nan, 
                         'F0': np.nan, 'In': None, 'F_fft': None, 
                         'x_fft': None, 'spectrum': None}
    else:
        if verbose>0:
            print('FT analysis on {0} datapoints ({8} to {9}): [{5}, {6}, ..., {7}] millisec (period: {1}, record frequency: {2}, {3} periods, {4} points per window)'.format(len(t_list), 
                            Period, FreqRecord, NumPeriods, PointsPerWindow, t_list[IndexEdges[0]]-t_list[0], t_list[IndexEdges[0]+1]-t_list[0], t_list[IndexEdges[1]-1]-t_list[0],
                            IndexEdges[0], IndexEdges[1]-1))

        # to get real amplitudes from FFT you have to divide by half the channel number
        PositionFFT = np.fft.rfft(x_list[IndexEdges[0]:IndexEdges[1]]) / PointsPerWindow * 2
        ForceFFT = np.fft.rfft(f_list[IndexEdges[0]:IndexEdges[1]]) / PointsPerWindow * 2
        AvgForce = np.mean(f_list[IndexEdges[0]:IndexEdges[1]])

        # process first harmonic to get moduli
        if NumPeriods < len(PositionFFT):
            PositionFFT_FirstHarm = PositionFFT[NumPeriods]
            ForceFFT_FirstHarm = ForceFFT[NumPeriods]
            G_FirstHarm = ForceFFT_FirstHarm/PositionFFT_FirstHarm
        else:
            PositionFFT_FirstHarm = np.nan
            ForceFFT_FirstHarm = np.nan
            G_FirstHarm = np.nan

        G_HigherHarm = []
        for j_harm in range(HigherHarmonics):
            #cur_pos_harm = PositionFFT[NumPeriods*j_harm]
            #cur_force_harm = ForceFFT[NumPeriods*j_harm]
            if NumPeriods*j_harm < len(ForceFFT) and NumPeriods < len(ForceFFT):
                G_HigherHarm.append(ForceFFT[NumPeriods*j_harm]/ForceFFT[NumPeriods])
            else:
                G_HigherHarm.append(np.nan)

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

        G_spectrum = []
        if return_spectrum:
            for i in range(len(PositionFFT)):
                G_spectrum.append(ForceFFT[i]/PositionFFT[i]) 

        return G_FirstHarm, {'F': ForceFFT_FirstHarm, 'x': PositionFFT_FirstHarm, 
                             'F0': AvgForce, 'In': G_HigherHarm, 'F_fft': ForceFFT, 
                             'x_fft': PositionFFT, 'spectrum': G_spectrum}