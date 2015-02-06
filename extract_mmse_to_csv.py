# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:31:13 2015

@author: mehdi.rahim@cea.fr
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fetch_data import set_fdg_pet_base_dir, set_rs_fmri_base_dir,\
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
DATA_DIR = set_rs_fmri_base_dir()
data = pd.read_csv(os.path.join(DATA_DIR, 'description_file.csv'))

 
no_mmse_subjects = []
n_mmse = 0
n_no_mmse = 0
mmses = []
for idx, row in data.iterrows():
    subject_id = row['Subject_ID']
    mmse = df[(df['PTID']==subject_id) & (df['VISCODE'] == 'sc')]\
             ['MMSCORE'].values
    
    if len(mmse) == 0:
        mmse = df[(df['PTID']==subject_id) & (df['VISCODE'] == 'v01')]\
                 ['MMSCORE'].values
    mmses.append(mmse[0])

data['MMSCORE'] = pd.Series(mmses, index=data.index)
data.to_csv(os.path.join(DATA_DIR, 'description_file_mmse.csv'))


    