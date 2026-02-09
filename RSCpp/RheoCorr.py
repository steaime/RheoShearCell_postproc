import os
import copy
import logging
import numpy as np
import scipy.interpolate as spi

def LoadForceCalib(FilePath, FitMethod='spline', FitParam=None, return_raw=False, use_cols=(0, 1), row_range=(1, -1), x_range=None):
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

    use_cols:   columns with position and force data

    row_range:  range of rows to be read. -1 denotes last row.

    x_range:    range of positions to be considered in the fitting procedure. If none, all data will be fitted
                has to be a tuple, (x_min, x_max)
                Set either value to None to disregard control
    """
    if os.path.isfile(FilePath):
        if row_range is None:
            row_range = (1, -1)
        if row_range[1] < row_range[0]:
            read_max_rows = None
        else:
            read_max_rows = row_range[1] - row_range[0]
        off, f = np.loadtxt(FilePath, skiprows=row_range[0], usecols=use_cols, unpack=True, max_rows=read_max_rows)
        if x_range is not None:
            if x_range[0] is None:
                x_range[0] = np.min(off)-1
            if x_range[1] is None:
                x_range[1] = np.max(off)+1
            mask = np.where(np.logical_and(off >= x_range[0], off <= x_range[1]))
            off, f = off[mask], f[mask]
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
    else:
        logging.error('Force calibration data file not found: ' + str(FilePath))
        return None

class ForceCorrection():

    def __init__(self, staticForce=None, forceCurrent=None):
        if isinstance(staticForce, dict):
            self.static_force = LoadForceCalib(**staticForce)
            if isinstance(self.static_force, tuple):
                self.static_force = self.static_force[0]
        else:
            self.static_force = staticForce
        if isinstance(forceCurrent, dict):
            self.force_current = LoadForceCalib(**forceCurrent)
            if isinstance(self.force_current, tuple):
                self.force_current = self.force_current[0]
        else:
            self.force_current = forceCurrent

    def Correct(self, force_data, pos_data, inplace=True):
        if inplace:
            res = force_data
        else:
            res = copy.deepcopy(force_data)
            
        if self.static_force is not None:
            res -= self.static_force(pos_data)
        if self.force_current is not None:
            res /= self.force_current(pos_data)
        
        return res