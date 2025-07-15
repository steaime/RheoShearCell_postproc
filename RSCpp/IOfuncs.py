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

def find_namebase(fpath, ret_suffix=False, rep_len=None, ext_len=4):
    if rep_len is None:
        rep_len = 1
    cur_fname = os.path.basename(fpath)[:-ext_len]
    suff = ''
    if cur_fname[-ext_len:] in ['_POS', '_NEG']:
        suff = cur_fname[-4-rep_len:]
        cur_fname = cur_fname[:-4]
    else: 
        suff = cur_fname[-rep_len:]
    logging.debug('filename processed to return filename (without extension) "{0}", namebase "{1}", suffix "{2}" (rep_len=={3})'.format(os.path.basename(fpath)[:-ext_len], cur_fname, suff, rep_len))
    if rep_len==0:
        ret_name = cur_fname
    else:
        ret_name = cur_fname[:-rep_len]
    if ret_suffix:
        return ret_name, suff
    else:
        return ret_name

def find_file_params(fname, explog_data, rep_len=None):
    if rep_len is None:
        rep_len = 1
    cur_namebase, suffix = find_namebase(fname, ret_suffix=True, rep_len=rep_len)
    find_params = explog_data[explog_data['Name'].str.contains(cur_namebase)]
    if len(find_params)==1:
        return find_params.iloc[0]
    elif len(find_params)>0:
        if suffix[-3:] == 'POS':
            return find_params.iloc[0]
        elif suffix[-3:] == 'NEG':
            return find_params.iloc[1]
        else:
            return find_params.iloc[0]
    else:
        logging.debug('find_params ({0}) has len==0'.format(cur_namebase))
        return None