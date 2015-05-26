# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:15:01 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob
import pandas as pd
import numpy as np
from datetime import date, timedelta



csv_file = 'Diagnosis/'
dataset_dir = '/disk4t/mehdi/data/ADNI_extracted/fmri_longitudinal/'


roster = pd.read_csv(os.path.join(csv_file, 'ROSTER.csv'))
dx = pd.read_csv(os.path.join(csv_file, 'DXSUM_PDXCONV_ADNIALL.csv'))

# DX_CURREN: 1=NL;2=MCI;3=AD
# DX_CONV : 1=Yes - Conversion;2=Yes - Reversion; 0=No
# DX_CONTYP 1=Normal Control to MCI; 2=Normal Control to AD; 3=MCI to AD
# DX_CHANGE: 1=Stable: NL to NL; 2=Stable: MCI to MCI; 
# 3=Stable: Dementia to Dementia; 4=Conversion: NL to MCI;
# 5=Conversion: MCI to Dementia; 6=Conversion: NL to Dementia;
# 7=Reversion: MCI to NL; 8=Reversion: Dementia to MCI;
# 9=Reversion: Dementia to NL
# DX_NORM, DX_MCI, DX_AD
# DX_ADES (1,2,3) : 1=Mild; 2=Moderate; 3=Severe
# DXAPP : 1=Probable; 2=Possible
#

def get_dx(rid, sid):
    """Returns all diagnoses for a given
    rid and sid"""
    
    # EXAMDATE
    dates = dx[(dx.RID == rid) & (dx.SITEID == sid)]['EXAMDATE'].values
    
    exam_date = []
    for d in dates:
        exam_date.append(date(int(d[:4]), int(d[5:7]), int(d[8:])))

    # DXCHANGE
    dxchange = dx[(dx.RID == rid) & (dx.SITEID == sid)]['DXCHANGE'].values
    dxchange = np.asarray(dxchange, dtype=int)
    
    return dxchange, exam_date



def get_acquisition_dates(subj_id):
    """Returns acquisition date from filename"""
    fnames = os.listdir(os.path.join(dataset_dir, subj_id, 'func'))
    
    dates = []
    for fname in fnames:
        d = fname.split('_')[-4]
#        t = '-'.join([d[:4], d[4:6], d[6:8]])
        t = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        dates.append(t)
    return sorted(dates)

def find_closest_exam_date(acq_date, exam_dates):
    """Returns closest date and indice of the
    closest exam_date from acq_date"""
    
    diff = []
    for e in exam_dates:
        diff.append(abs(acq_date - e))
    ind = np.argmin(diff)
    print diff[ind]
    return exam_dates[ind]


dx_list = np.array(['None',
                    'Normal',
                    'MCI',
                    'AD',
                    'Normal->MCI',
                    'MCI->AD',
                    'Normal->AD',
                    'MCI->Normal',
                    'AD->MCI',
                    'AD->Normal'])

subj_list = os.listdir(dataset_dir)
for subj_id in subj_list:
    rid = roster[roster.PTID == subj_id]['RID'].values
    sid = roster[roster.PTID == subj_id]['SITEID'].values
    if len(rid) > 0 and len(sid) > 0:
        rid = rid[0]
        sid = sid[0]
        dxs, ex_dates = get_dx(rid, sid)
        acq_dates = get_acquisition_dates(subj_id)
        print subj_id, acq_dates[0], find_closest_exam_date(acq_dates[0], ex_dates)
        print dx_list[dxs]
        print '-'
    else:
        print 'something is missing'



