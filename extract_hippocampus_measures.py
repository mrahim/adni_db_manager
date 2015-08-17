# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:58:34 2015

@author: mr243268
"""

import os
import numpy as np
import pandas as pd
from datetime import date
from fetch_data import set_group_indices


def get_dx(rid, exam=None):
    """Returns all diagnoses for a given
    rid and sid"""

    dates = dx[dx.RID == rid]['EXAMDATE'].values
    exam_dates = [date(int(d[:4]), int(d[5:7]), int(d[8:])) for d in dates]


    # DXCHANGE
    change = dx[dx.RID == rid]['DXCHANGE'].values
    curren = dx[dx.RID == rid]['DXCURREN'].values

    # change, curren have the same length
    dxchange = [int(np.nanmax([change[k], curren[k]])) for k in range(len(curren))]

    if exam is not None and len(exam_dates) > 0:
        exam_date, ind = find_closest_exam_date(exam, exam_dates)
        ###TODO : return exam_date?
        return dxchange[ind]
    else:
        return -4


def find_closest_exam_date(acq_date, exam_dates):
    """Returns closest date and indice of the
    closest exam_date from acq_date"""
    
    diff = [abs(acq_date - e) for e in exam_dates]
    ind = np.argmin(diff)
    return exam_dates[ind], ind

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

roster = pd.read_csv('Diagnosis/ROSTER.csv')
dx = pd.read_csv('Diagnosis/DXSUM_PDXCONV_ADNIALL.csv')
dob = pd.read_csv('Diagnosis/PTDEMOG.csv')
fs = pd.read_csv('Diagnosis/UCSFFSX51_05_20_15.csv')
fs = fs.drop_duplicates('RID')

rids = fs['RID'].values
exams = map(lambda e: date(int(e[:4]),
                           int(e[5:7]),
                           int(e[8:])), fs['EXAMDATE'].values)

# Extract diagnosis
dxs = np.array(map(get_dx, rids, exams))
dx_group = dx_list[dxs]
dx_ = set_group_indices(dx_group)


# Extract hippocampus values
column_idx = np.arange(131, 147)
cols = ['ST' + str(c) + 'HS' for c in column_idx]
X = fs[cols].values

# Show data
idx_ = np.hstack((dx_['AD'], dx_['Normal']))
x = X[idx_, :]
y = np.array([1]*len(dx_['AD']) + [0]*len(dx_['Normal']))

# Remove nans
idx = np.array([ ~np.isnan(v).all() for v in x])
x = x[idx]
y = y[idx]


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2, random_state=42)
lr = LogisticRegression()

acc = []
for train, test in sss:
    lr.fit(x[train], y[train])
    acc.append(lr.score(x[test], y[test])) 

print np.mean(acc)


