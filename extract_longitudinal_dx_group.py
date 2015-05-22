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
    return dx[(dx.RID == rid) & (dx.SITEID == sid)][['EXAMDATE', 'DXCURREN', 'DXCHANGE']].values




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
    break