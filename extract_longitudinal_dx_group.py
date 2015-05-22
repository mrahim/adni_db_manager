# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:15:01 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob
import pandas as pd
import numpy as np

csv_file = 'Diagnosis/'
dataset_dir = '/disk4t/mehdi/data/ADNI_extracted/fmri_longitudinal/'


roster = pd.read_csv(os.path.join(csv_file, 'ROSTER.csv'))
dx = pd.read_csv(os.path.join(csv_file, 'DXSUM_PDXCONV_ADNIALL.csv'))


def get_dx(rid, sid):
    """Returns all diagnoses for a given
    rid and sid"""
    return dx[(dx.RID == rid) & (dx.SITEID == sid)][['EXAMDATE', 'DX_CURREN', 'DX_CHANGE']].values




subj_list = os.listdir(dataset_dir)
for subj_id in subj_list:
    rid = roster[roster.PTID == subj_id]['RID'].values
    sid = roster[roster.PTID == subj_id]['SITEID'].values
    if len(rid) > 0 and len(sid) > 0:
        rid = rid[0]
        sid = sid[0]
        print sid, rid, get_dx(rid, sid)
    else:
        print 'something is missing'
