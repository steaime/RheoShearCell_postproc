import bisect
import logging
import numpy as np
import matplotlib.pyplot as plt

from rDLSpp import FTrheo as ft
from rDLSpp import IOfuncs as iof

from DSH import SharedFunctions as sf

def proc_file(fpath, explog_data, rep_len=1, anal_type='read', anal_params={}, usecols=None):
    find_params = iof.find_file_params(fpath, explog_data, rep_len=rep_len)
    if find_params is not None:
        if usecols is None:
            if int(find_params['Axis'])==1:
                cur_straincol, cur_osrstrain_col = 5, 2
            else:
                cur_straincol, cur_osrstrain_col = 2, 5
            usecols=(1, cur_straincol, 6)
        if anal_type in ['read', 'plot', 'flowcurve', 'avgperiod']:
            t, strain, stress = iof.ReadRheoData(fpath, usecols=usecols, unpack=True)
        if anal_type=='count':
            return 1
        elif anal_type=='read':
            return t, strain, stress
        elif anal_type=='plot':
            ax = anal_params['ax']
            if anal_params['plot_type'] == 'time':
                ax[0].plot((t-t[0])/1000-anal_params['t0'], strain, anal_params['fmt'])
                ax[1].plot((t-t[0])/1000-anal_params['t0'], stress, anal_params['fmt'])
            elif anal_params['plot_type'] == 'stressstrain':
                ax.plot(strain, stress, anal_params['fmt'])
            elif anal_params['plot_type'] == 'stressrelax':
                ax.plot((t-t[0])/1000-anal_params['t0'], stress, anal_params['fmt'])
        elif anal_type == 'avgperiod':
            if 'StartIdx' not in anal_params:
                anal_params['StartIdx'] = 0
            if 'EndIdx' not in anal_params:
                anal_params['EndIdx'] = 0
            avg_res = np.empty((anal_params['PeriodIdx'], 3), dtype=float)
            avg_res[:,0] = (t[:anal_params['PeriodIdx']]-t[0])/1000
            for i in range(avg_res.shape[0]):
                avg_res[i,1] = np.mean(strain[i::anal_params['PeriodIdx']])
                avg_res[i,2] = np.mean(stress[i::anal_params['PeriodIdx']])
            return avg_res
        elif anal_type in ['FT', 'OSR']:
            if 'Tres_Step' in anal_params and 'Tres_Nint' not in anal_params:
                anal_params['Tres_Nint'] = 1
            OSR_period, ORS_amp = None, None
            if 'OSRparams' in anal_params:
                if anal_params['OSRparams'] is not None:
                    OSR_period = anal_params['OSRparams']['Period']
                    ORS_amp = anal_params['OSRparams']['Amp']
            tres_res = None
            if anal_type=='FT':
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
                t = iof.ReadRheoData(fpath, usecols=(1), unpack=True)
                int_duration = (t[-1]-t[0])/1000
                if OSR_period is not None:
                    if 'nperiods' not in anal_params:
                        anal_params['nperiods'] = int(int_duration/OSR_period - 2)
                    G, opt = ft.FTanalysisRheology(fpath, Period=OSR_period, AnalyzePeriods=anal_params['nperiods'], 
                                                 FreqRecord=None, usecols=(1,cur_osrstrain_col,6))
                    if 'Tres_Step' in anal_params:
                        tres_res = ft.CalcTimeDependentModuli(fpath, Period=OSR_period, StepTime=anal_params['Tres_Step'], 
                                                           AnalyzePeriods=anal_params['Tres_Nint'], Duration=int_duration, usecols=(1,cur_osrstrain_col,6))
                else:
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
        if anal_type=='count':
            return 0
        else:
            return None

def proc_files(fpath_list, explog_data, filter_type=None, filter_axis=None, filter_name=None, rep_len=1, max_num=None, usecols=None, anal_type='read', anal_params={}):
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
            if do_process and filter_axis is not None:
                do_process = (int(find_params['Axis']) == filter_axis)
            if do_process and filter_name is not None:
                do_process = (filter_name in find_params['Name'])
            if do_process:
                logging.debug('Processing file {0}/{1}, of type {2} (fname: {3})'.format(i, len(fpath_list), find_params['Type'], cur_fname))
                if anal_type=='OSR':
                    if 'OSRparam_list' in anal_params:
                        anal_params['OSRparams'] = anal_params['OSRparam_list'][i]
                res.append(proc_file(fpath_list[i], explog_data, anal_type=anal_type, anal_params=anal_params, usecols=usecols))
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
        res_arr = np.asarray([[x['StartedOn'], 
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
                               x['OSR_Amp'] 
                              ] 
                              for x in res])
        return res_arr
    elif anal_type=='flowcurve':
        res_arr = np.asarray(res)
        return res_arr[:,0], res_arr[:,1]
    else:
        return res