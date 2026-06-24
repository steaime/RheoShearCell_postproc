import os
import bisect
import logging
import numpy as np
import matplotlib.pyplot as plt

from RSCpp import FTrheo as ft
from RSCpp import IOfuncs as iof
from RSCpp import SharedFunctions as sf

def proc_file(fpath, explog_data, rep_len=None, anal_type='read', anal_params={}, usecols=None, decimate=0):
    logging.debug('Now processing file ' + fpath)
    find_params = iof.find_file_params(fpath, explog_data, rep_len=rep_len)
    #logging.debug('proc_file procedure extracted parameters file {0}: {1}'.format(fpath, find_params))
    if find_params is not None:
        if usecols is None:
            if int(find_params['Axis'])==1:
                cur_straincol, cur_osrstrain_col = 5, 2
            else:
                cur_straincol, cur_osrstrain_col = 2, 5
            usecols=(1, cur_straincol, 6)
        if anal_type in ['read', 'plot', 'flowcurve', 'avgperiod']:
            t, strain, stress = iof.ReadRheoData(fpath, usecols=usecols, unpack=True, decimate=decimate)
            if 'print_fnames' in anal_params:
                if anal_params['print_fnames']:
                    print(os.path.basename(fpath))
        if anal_type=='count':
            return 1
        elif anal_type=='read':
            return t, strain, stress
        elif anal_type=='plot':
            ax = anal_params['ax']
            t_plot, strain_plot, stress_plot = t, strain, stress
            if 'plot_slice' in anal_params:
                if anal_params['plot_slice'] is not None:
                    slc = slice(*anal_params['plot_slice'])
                    t_plot, strain_plot, stress_plot = t[slc], strain[slc], stress[slc]
            if anal_params['plot_type'] == 'time':
                if 'global_time' in anal_params:
                    if anal_params['global_time'] == 'auto':
                        anal_params['global_time'] = t[0]
                    t_off = anal_params['global_time']
                else:
                    t_off = t[0]
                if t_plot is not None and strain_plot is not None and stress_plot is not None:
                    plot_data = [(t_plot-t_off)/1000-anal_params['t0'], strain_plot, stress_plot]
                    ax[0].plot(plot_data[0], plot_data[1], anal_params['fmt'])
                    ax[1].plot(plot_data[0], plot_data[2], anal_params['fmt'])
                else:
                    logging.warn('Error plotting content of file ' + fpath)
                    plot_data = None
                return plot_data
            elif anal_params['plot_type'] == 'stressstrain':
                if 'stress_corr_spl' in anal_params:
                    stress_plot = stress_plot - anal_params['stress_corr_spl'](strain_plot)
                if 'strain_off' in anal_params:
                    if anal_params['strain_off'] == 'first':
                        strain_plot = strain_plot - strain[0]
                    else:
                        strain_plot = strain_plot - anal_params['strain_off']
                if 'strain_abs' in anal_params:
                    if anal_params['strain_abs']:
                        strain_plot = np.abs(strain_plot)
                if 'stress_abs' in anal_params:
                    if anal_params['stress_abs']:
                        stress_plot = np.abs(stress_plot)
                ax.plot(strain_plot, stress_plot, anal_params['fmt'])
                return [strain_plot, stress_plot]
            elif anal_params['plot_type'] == 'stressrelax':
                ax.plot((t_plot-t[0])/1000-anal_params['t0'], stress_plot, anal_params['fmt'])
                return [(t_plot-t[0])/1000-anal_params['t0'], stress_plot]
        elif anal_type == 'avgperiod':
            if 'StartIdx' not in anal_params:
                anal_params['StartIdx'] = 0
            if 'EndIdx' not in anal_params:
                anal_params['EndIdx'] = 0
            avg_res = np.empty((anal_params['PeriodIdx'], 3), dtype=float)
            avg_res[:,0] = (t[anal_params['StartIdx']:anal_params['StartIdx']+anal_params['PeriodIdx']]-t[0])/1000
            for i in range(avg_res.shape[0]):
                avg_res[i,1] = np.mean(strain[anal_params['StartIdx']+i:len(t)-anal_params['EndIdx']:anal_params['PeriodIdx']])
                avg_res[i,2] = np.mean(stress[anal_params['StartIdx']+i:len(t)-anal_params['EndIdx']:anal_params['PeriodIdx']])
            return avg_res
        elif anal_type in ['FT', 'OSR']:
            if 'Tres_Step' in anal_params and 'Tres_Nint' not in anal_params:
                anal_params['Tres_Nint'] = 1
            OSR_period, ORS_amp = None, None
            if 'OSRparams' in anal_params:
                if anal_params['OSRparams'] is not None:
                    OSR_period = anal_params['OSRparams']['Period']
                    ORS_amp = anal_params['OSRparams']['Amp']
                    logging.debug('OSR period ({0:.2f}) and amplitude ({1:.3f}) read from analysis parameters'.format(OSR_period, ORS_amp))
            tres_res = None
            if anal_type=='FT':
                logging.debug('FT analysis')
                OSR_period, ORS_amp = np.nan, np.nan
                int_duration = float(find_params['Duration'])
                int_period = float(find_params['Period'])
                if 'nperiods' not in anal_params:
                    anal_params['nperiods'] = int(int_duration/int_period - 2)
                G, opt = ft.FTanalysisRheology(fpath, Period=int_period, AnalyzePeriods=anal_params['nperiods'], 
                                             FreqRecord=None, usecols=(1,cur_straincol,6))
                if 'Tres_Step' in anal_params:
                    tres_res = ft.CalcTimeDependentModuli(fpath, Period=int_period, StepTime=anal_params['Tres_Step'], 
                                                      AnalyzePeriods=anal_params['Tres_Nint'], Duration=int_duration, usecols=(1,cur_straincol,6))
                    
            else:
                logging.debug('OSR analysis')
                t = iof.ReadRheoData(fpath, usecols=(1), unpack=True)
                int_duration = (t[-1]-t[0])/1000
                if OSR_period is not None:
                    if 'nperiods' not in anal_params:
                        anal_params['nperiods'] = int(int_duration/OSR_period - 2)
                        logging.debug('Number of periods automatically set to {0}'.format(anal_params['nperiods']))
                    else:
                        logging.debug('Number of periods manually set to {0} (OSR period: {1:.1f}ms)'.format(anal_params['nperiods'], OSR_period))
                    G, opt = ft.FTanalysisRheology(fpath, Period=OSR_period, AnalyzePeriods=anal_params['nperiods'], 
                                                 FreqRecord=None, usecols=(1,cur_osrstrain_col,6))
                    if 'Tres_Step' in anal_params:
                        tres_res = ft.CalcTimeDependentModuli(fpath, Period=OSR_period, StepTime=anal_params['Tres_Step'], 
                                                           AnalyzePeriods=anal_params['Tres_Nint'], Duration=int_duration, usecols=(1,cur_osrstrain_col,6))
                else:
                    logging.debug('OSR_period needed for analysis of file ' + fpath)
                    opt = None
            if tres_res is not None:
                tres_fpath = sf.AddSuffixToPath(fpath, '_tres' + anal_type)
                np.savetxt(tres_fpath, tres_res, delimiter='\t', header='t[s]\tGp\tGs')
            if opt is not None:
                opt['OSR_Amp'] = ORS_amp
                opt['OSR_Period'] = OSR_period
                opt['Type'] = int(find_params['Type'])
                opt['StrainControlled'] = (find_params['StrainControlled'] == '1')
                if opt['StrainControlled']:
                    opt['Amplitude'] = float(find_params['Displacement'])
                else:
                    opt['Amplitude'] = float(find_params['Force'])
                try:
                    opt['Period'] = float(find_params['Period'])
                except:
                    opt['Period'] = np.nan
                try:
                    opt['Offset'] = float(find_params['Offset'])
                except:
                    opt['Offset'] = np.nan
                opt['StartedOn'] = float(find_params['StartedOn'])
            return opt
        elif anal_type=='flowcurve':
            d_range = anal_params['displ_range']
            min_idx = bisect.bisect(np.abs(strain-strain[0]), d_range[0])
            max_idx = bisect.bisect(np.abs(strain-strain[0]), d_range[1])
            if max_idx <= min_idx:
                raise ValueError('Error bisecting strain list (start: {0}, end: {1}) with displacement range {2}'.format(strain[0], strain[1], d_range))
                return None
            else:
                return float(find_params['Speed']), np.mean(stress[min_idx:max_idx])
    else:
        logging.warning('proc_file procedure failed on file {0}: no parameters found'.format(fpath))
        if anal_type=='count':
            return 0
        else:
            return None

def proc_files(fpath_list, explog_data, filter_type=None, exclude_type=None, filter_axis=None, filter_name=None, rep_len=None, max_num=None, usecols=None, decimate=0, anal_type='count', anal_params={}):
    rep_count = None
    if rep_len is None:
        for test_len in range(6):
            rep_count = proc_files(fpath_list, explog_data, filter_type=filter_type, filter_axis=filter_axis, filter_name=filter_name, 
                                   rep_len=test_len, max_num=max_num, usecols=usecols, anal_type='count', anal_params=anal_params)
            if rep_count>0:
                rep_len=test_len
                break
        if rep_count>0:
            logging.info('Auto-detected suffix length: {0} ({1} files detected)'.format(rep_len, rep_count))
        else:
            logging.warn('Impossible to auto-detect suffix length')
    res = []
    anal_params = anal_params
    proc_count = 0
    if anal_type=='plot':
        cycle_num = int(proc_files(fpath_list, explog_data, filter_type=filter_type, filter_axis=filter_axis, filter_name=filter_name, rep_len=rep_len, anal_type='count', max_num=max_num, usecols=usecols))
        if cycle_num<=0:
            cycle_num=1
        if 'plot_type' not in anal_params:
            anal_params['plot_type'] = 'time'
        if 't0' not in anal_params:
            anal_params['t0'] = 0.0
        if 'fmt' not in anal_params:
            anal_params['fmt'] = '-'
        fig, ax = plt.subplots()
        ax.set_prop_cycle(color=plt.cm.cool(np.linspace(0,1,cycle_num)))
        if anal_params['plot_type'] == 'time':
            ax2 = ax.twinx()
            ax2.set_prop_cycle(color=plt.cm.summer(np.linspace(0,1,cycle_num)))
            ax.set_xlabel('time [s]')
            ax.set_ylabel('displacement [mm]')
            ax2.set_ylabel('force [N]')
            anal_params['ax'] = [ax, ax2]
        elif anal_params['plot_type'] == 'stressstrain':
            ax.set_xlabel('displacement [mm]')
            ax.set_ylabel('force [N]')
            anal_params['ax'] = ax
        elif anal_params['plot_type'] == 'stressrelax':
            ax.set_xlabel('time [s]')
            ax.set_ylabel('force [N]')
            ax.set_xscale('log')
            anal_params['ax'] = ax
    for i in range(len(fpath_list)):
        cur_fname = fpath_list[i]
        find_params = iof.find_file_params(cur_fname, explog_data, rep_len=rep_len)
        if find_params is None:
            logging.debug('Skipping file {0}/{1}: no match found in parameter table for filename "{2}" (namebase: {3})'.format(i, len(fpath_list), cur_fname, iof.find_namebase(cur_fname, rep_len=rep_len)))
        else:
            do_process = True
            if filter_type is not None:
                do_process = (int(find_params['Type']) == filter_type)
            if do_process and exclude_type is not None:
                do_process = (int(find_params['Type']) != exclude_type)
            if do_process and filter_axis is not None:
                do_process = (int(find_params['Axis']) == filter_axis)
            if do_process and filter_name is not None:
                do_process = (filter_name in find_params['Name'])
            if do_process:
                logging.debug('Processing file {0}/{1}, of type {2} (Name: {3}; fname: {4}). Analysis type: {5}'.format(i, len(fpath_list), find_params['Type'], find_params['Name'], cur_fname, anal_type))
                if anal_type=='OSR':
                    if 'OSRparam_list' in anal_params:
                        anal_params['OSRparams'] = anal_params['OSRparam_list'][i]
                        logging.debug('OSRparams extracted from list: {0}'.format(anal_params['OSRparams']))
                res.append(proc_file(fpath_list[i], explog_data, anal_type=anal_type, anal_params=anal_params, usecols=usecols, decimate=decimate, rep_len=rep_len))
                if max_num is not None:
                    if len(res) >= max_num:
                        logging.warn('[{0}/{1}] : reached limit ({2}) of files to be processed'.format(i, len(fpath_list), max_num))
                        break
            else:
                logging.debug('[{0}/{1}] : Skipping file {2} due to imposed filter (Type: {3}, Axis: {4}, Name: {5})'.format(i, len(fpath_list), cur_fname, find_params['Type'], find_params['Axis'], find_params['Name']))
    #if anal_type=='read':
    #    res_arr = np.asarray(res)
    #    return res_arr
    if anal_type=='plot':
        if 'return_data' in anal_params:
            if anal_params['return_data']:
                return res
        return fig
    elif anal_type=='count':
        return np.sum(res)
    elif anal_type=='avgperiod':
        if len(res) > 0:
            res_comb = np.empty((res[0].shape[0], 1+2*len(res)), dtype=float)
            res_comb[:,0] = res[0][:,0]
            for i in range(len(res)):
                res_comb[:,1+2*i] = res[i][:,1]
                res_comb[:,1+2*i+1] = res[i][:,2]
            return res_comb
        else:
            return None
    elif anal_type in ['FT', 'OSR']:
        res_list = []
        for x in res:
            if x is None:
                res_list.append([np.nan]*15)
            else:
                res_list.append([x['StartedOn'], 
                            x['Amplitude'], 
                            x['Period'], 
                            x['Offset'], 
                            np.abs(x['F']), 
                            np.angle(x['F']), 
                            np.abs(x['x']), 
                            np.angle(x['x']), 
                            np.abs(x['F']/x['x']), 
                            -np.angle(x['F']/x['x']), 
                            -np.real(x['F']/x['x']), 
                            -np.imag(x['F']/x['x']), 
                            x['F0'],
                            x['OSR_Period'], 
                            x['OSR_Amp']])
        res_arr = np.asarray(res_list)
        return res_arr
    elif anal_type=='flowcurve':
        res_arr = np.asarray(res)
        return res_arr[:,0], res_arr[:,1]
    else:
        return res
    

def BatchProcess(explog, filter_type=None, exclude_type=None, filter_axis=None, filter_name=None, rep_len=None, max_num=None, usecols=None, decimate=0, anal_type='read', anal_params={}):
    fpath_list = list(explog['FilePath'])
    if len(fpath_list) > 0:
        return proc_files(fpath_list, explog, filter_type=filter_type, exclude_type=exclude_type, filter_axis=filter_axis, filter_name=filter_name, rep_len=rep_len, max_num=max_num, usecols=usecols, decimate=decimate, anal_type=anal_type, anal_params=anal_params)