import os
import numpy as np
import logging


def ReadRheoData(fname, usecols=(1,2,6), unpack=True, **loadtxt_kwargs):
    """Reads a text file with output of a rheology experiment

    Parameters
    ----------
    fname :     full path of the filename with rheology data
    usecols :   select the columns that should be read.
                the file structure is: {#; Time; Position; Speed; Position error; Force}
                the default parameter (1,2,5) corresponds to reading time, position and force
    unpack :    if True, it will return each column as a separate 1D array
                otherwise, it will return a 2D array
    loadtxt_kwargs : other kwargs to be passed to np.loadtxt
    """
    if os.path.isfile(fname):
        if usecols is None:
            strlog = 'Reading all columns from file ' + str(fname)
        else:
            strlog = 'Reading columns ' + str(usecols) + ' from file ' + str(fname)
        if unpack:
            strlog += ' (unpack)'
        logging.debug(strlog)
        return np.loadtxt(fname, delimiter='\t', usecols=usecols, unpack=unpack, **loadtxt_kwargs)
    else:
        logging.error('ReadData error: file ' + str(fname) + ' does not exist')
        if unpack and len(usecols)>1:
            return [None]*len(usecols)
        else:
            return None

    