# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:31:13 2015

@author: mehdi.rahim@cea.fr
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet,\
                       set_features_base_dir

FEAT_DIR = set_features_base_dir()

# Merge Roster with MMSE csv in order to match subject_id with mmse
def merge_adni_csv():
    mmse_file_path = os.path.join(FEAT_DIR, 'Assessments', 'MMSE.csv')
    roster_file_path = os.path.join(FEAT_DIR, 'Enrollment', 'ROSTER.csv')
    mmse_df = pd.read_csv(mmse_file_path)
    roster_df = pd.read_csv(roster_file_path)
    merged_df = pd.merge(roster_df, mmse_df, how='inner', on=['RID', 'SITEID'])
    merged_df.to_csv(os.path.join(FEAT_DIR, 'Assessments', 'MMSE_merged.csv'))

# Load MMSE
mmse_file_path = os.path.join(FEAT_DIR, 'Assessments', 'MMSE_merged.csv')
df = pd.read_csv(mmse_file_path)

# Load Dataset
dataset = fetch_adni_petmr()

no_mmse_subjects = []
n_mmse = 0
n_no_mmse = 0
mmmse = []
for subject_id, idx in zip(dataset['subjects'],
                           range(len(dataset['subjects']))):
    mmse = df[df['PTID']==subject_id]['MMSCORE'].values    
    
    print int(np.mean(mmse)), int(np.std(mmse)), dataset['dx_group'][idx]
    if np.std(mmse) > 3:
        print mmse
    if np.all(np.isnan(mmse)):
        no_mmse_subjects.append(subject_id)
        n_no_mmse += 1
    else:
        n_mmse += 1
        mmmse.append(str(int(np.mean(mmse))))
    