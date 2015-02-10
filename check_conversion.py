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


def merge_adni_csv(csv_file):

    root, _ = os.path.splitext(csv_file)    
    
    mmse_file_path = os.path.join(FEAT_DIR, 'Assessments', csv_file)
    roster_file_path = os.path.join(FEAT_DIR, 'Enrollment', 'ROSTER.csv')
    mmse_df = pd.read_csv(mmse_file_path)
    roster_df = pd.read_csv(roster_file_path)
    merged_df = pd.merge(roster_df, mmse_df, how='inner', on=['RID', 'SITEID'])
    merged_df.to_csv(os.path.join(FEAT_DIR, 'Assessments',
                                  root + '_merged.csv'))


# Load MMSE
mmse_file_path = os.path.join(FEAT_DIR, 'Assessments',
                              'DXSUM_PDXCONV_ADNIALL_corr_merged.csv')
df = pd.read_csv(mmse_file_path)

subjects = df['PTID'].unique() 

all_scores = []
converters = []


### DXCHANGE == ADNI 2, G0
cpt_adni2 = 0
for subject_id in subjects:
    dx_change = df[df['PTID'] == subject_id]['DXCHANGE'].values
    visit = df[df['PTID'] == subject_id]['VISCODE2'].values
    phase = df[df['PTID'] == subject_id]['Phase_x'].values
    dx_change = dx_change[~np.isnan(dx_change)]

    if len(dx_change) > 0:
        dx_change = np.delete(dx_change, np.where(dx_change == -4))
        all_scores.append(dx_change)
        if np.std(dx_change) > 0:
            #print subject_id, dx_change, phase
            if (2 in dx_change or 5 in dx_change) and (3 in dx_change): 
                converters.append(subject_id)
                print subject_id, dx_change, visit, phase
                cpt_adni2 += 1

adni2 = np.array(converters, copy=True)

adni1 = []
cpt_adni1 = 0
### DXCURREN == ADNI 1, GO
for subject_id in subjects:
    dx_curren = df[df['PTID'] == subject_id]['DXCURREN'].values
    visit = df[df['PTID'] == subject_id]['VISCODE2'].values
    phase = df[df['PTID'] == subject_id]['Phase_x'].values
    dx_curren = dx_curren[~np.isnan(dx_curren)]
    if len(dx_curren) > 0:
        if (2 in dx_curren or 5 in dx_curren) and (3 in dx_curren):
            print subject_id, dx_curren, visit, phase
            converters.append(subject_id)
            adni1.append(subject_id)
            cpt_adni1 += 1

"""
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
"""