import os
import numpy as np
import scipy.interpolate as spi
import logging
from DSH import SharedFunctions as sf


def ReadData(fname, usecols=None, unpack=True, delimiter='\t', **loadtxt_kwargs):
    if os.path.isfile(fname):
        if usecols is None:
            strlog = 'Reading all columns from file ' + str(fname)
        else:
            strlog = 'Reading columns ' + str(usecols) + ' from file ' + str(fname)
        if unpack:
            strlog += ' (unpack)'
        logging.info(strlog)
        return np.loadtxt(fname, delimiter=delimiter, usecols=usecols, unpack=unpack, **loadtxt_kwargs)
    else:
        logging.error('ReadData error: file ' + str(fname) + ' does not exist')
        return None

def LoadForceCalib(FilePath, FitMethod='spline', FitParam=None, return_raw=False):
    """Loads calibration data for current-force conversion

    Parameters
    ----------
    FilePath :  full path of the filename with calibration data
    FitMethod : {'spline'|'poly'}, method for fitting the raw calibration data
    FitParam :  float, parameter to pass to fitting method.
                - for spline fitting, FitParam will be the 's' parameter of the spline fitting
                  (the smaller s, the larger number of nodes)
                - for poly fitting, FitParam will be the degree of the polynomial fit
                  (if None, cubic fitting will be used by default)
    return_raw: Boolean. If true, return the raw calibration data together with the fit function
    """
    off, f = np.loadtxt(FilePath, skiprows=1, unpack=True)
    if FitMethod=='spline':
        fitres = spi.UnivariateSpline(off, f, s=FitParam)
    elif FitMethod=='poly':
        if FitParam is None:
            FitParam = 3
        fitres = np.poly1d(np.polyfit(off, f, FitParam))
    if return_raw:
        return fitres, [off, f]
    else:
        return fitres