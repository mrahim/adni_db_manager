# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 17:18:31 2015

@author: mehdi.rahim@cea.fr
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fetch_data import set_fdg_pet_base_dir, set_rs_fmri_base_dir,\
                       set_features_base_dir, fetch_adni_petmr

FEAT_DIR = set_features_base_dir()

# Merge Roster with MMSE csv in order to match subject_id with mmse
def merge_adni_csv():
    csf_file_path = os.path.join(FEAT_DIR, 'CSF', 'LOCLAB.csv')
    roster_file_path = os.path.join(FEAT_DIR, 'Enrollment', 'ROSTER.csv')
    csf_df = pd.read_csv(csf_file_path)
    roster_df = pd.read_csv(roster_file_path)
    merged_df = pd.merge(roster_df, csf_df, how='inner', on=['RID'])
    merged_df.to_csv(os.path.join(FEAT_DIR, 'CSF', 'CSF_merged_3.csv'))



filenames = ['CSF_merged.csv', 'CSF_merged_2.csv', 'CSF_merged_3.csv']


S = []
D = []
for fname in filenames:
    # Load CSF
    csf_file_path = os.path.join(FEAT_DIR, 'CSF', fname)
    df = pd.read_csv(csf_file_path)
    
    
    dataset = fetch_adni_petmr()
    s = []
    d = []
    for i in range(len(dataset['subjects'])):
        if len(df[df['PTID'] == dataset['subjects'][i]])>0:
            s.append(dataset['subjects'][i])
            d.append(dataset['dx_group'][i])
    print len(s)
    S.append(s)
    D.append(d)
