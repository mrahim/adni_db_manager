# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:15:01 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob, shutil
import pandas as pd
import numpy as np
<<<<<<< HEAD
from datetime import date, timedelta
=======
from datetime import date
>>>>>>> 42c1aa655f953d0bb3cb81f4888a24abd5c3d319

csv_file = 'Diagnosis/'
dataset_dir = '/disk4t/mehdi/data/ADNI_extracted/fmri_longitudinal/'

<<<<<<< HEAD
=======
tobepreprocessed_dir = '/disk4t/mehdi/data/ADNI_extracted/' \
                        'fmri_longitudinal_acquisitions/'


>>>>>>> 42c1aa655f953d0bb3cb81f4888a24abd5c3d319
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


<<<<<<< HEAD
def get_acquisition_dates(subj_id):
=======

def get_acquisition_data(subj_id):
>>>>>>> 42c1aa655f953d0bb3cb81f4888a24abd5c3d319
    """Returns acquisition date from filename"""
    fnames = os.listdir(os.path.join(dataset_dir, subj_id, 'func'))
    
    dates = []
    for fname in fnames:
        d = fname.split('_')[-4]
        t = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        dates.append(t)
    return dates, fnames

def find_closest_exam_date(acq_date, exam_dates):
    """Returns closest date and indice of the
    closest exam_date from acq_date"""
    
    diff = []
    for e in exam_dates:
        diff.append(abs(acq_date - e))
    ind = np.argmin(diff)
    return exam_dates[ind], ind


def is_stable(dxchange):
    """Checks if dx are stable
    """
    for a in dxchange[:-1]:
        for b in dxchange[1:]:
            if a != b:
                return False
    return True


def get_image_id(fname):
    """Returns img_id
    """
    return fname.split('_')[-1][:-4]


###############################################################################
# Main loop
###############################################################################
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

subj_dict_list = []
subj_list = os.listdir(dataset_dir)
for subj_id in subj_list:
    rid = roster[roster.PTID == subj_id]['RID'].values
    sid = roster[roster.PTID == subj_id]['SITEID'].values
    if len(rid) > 0 and len(sid) > 0:
        rid = rid[0]
        sid = sid[0]
        dxs, ex_dates = get_dx(rid, sid)
        acq_dates, acq_names = get_acquisition_data(subj_id)
        if len(dxs) > 0 and len(acq_dates) > 0:
            for i, acq_date in enumerate(acq_dates):
                exam_date, exam_ind = find_closest_exam_date(acq_date, ex_dates)
                if dxs[exam_ind] < 4: # NO CONVERT OR REVERT !!!
                    subj_dict = {}
                    subj_dict['Subject ID'] = subj_id
                    subj_dict['DX Group'] = dx_list[dxs[exam_ind]]
                    subj_dict['EXAM_DATE'] = exam_date
                    subj_dict['ACQ_DATE'] = acq_date
                    subj_dict['FILENAME'] = acq_names[i]
                    #subj_dict['DIFF_DATE'] = acq_date - exam_date
                    #subj_dict['STABLE'] = is_stable(dxs)
                    subj_dict_list.append(subj_dict)
    else:
        print 'something is missing'

df = pd.DataFrame.from_dict(subj_dict_list)
df = df[['Subject ID', 'DX Group', 'ACQ_DATE', 'EXAM_DATE', 'FILENAME']]


#for idx, row in df.iterrows():
#    id_img = get_image_id(row['FILENAME'])
#    dst_dir = os.path.join(tobepreprocessed_dir, id_img)
#    dst_file = os.path.join(dst_dir, row['FILENAME'])
#    src_file = os.path.join(dataset_dir, row['Subject ID'], 'func', row['FILENAME'])
#    if os.path.isfile(src_file):
#        if not os.path.isdir(dst_dir):
#            os.mkdir(dst_dir)
#        shutil.copy(src_file, dst_file)
#        print src_file, dst_dir
#
#
