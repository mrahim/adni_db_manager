# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:31:13 2015

@author: mehdi.rahim@cea.fr
"""
import os
import pandas as pd
import numpy as np
from fetch_data import fetch_adni_petmr, fetch_adni_fdg_pet, set_features_base_dir


FEAT_DIR = set_features_base_dir()

mmse_file_path = os.path.join(FEAT_DIR, 'Assessments', 'ADAS_ADNIGO2.csv')

#dataset = fetch_adni_petmr()
dataset = fetch_adni_fdg_pet()

df = pd.read_csv(mmse_file_path)


for subject_id in dataset['subjects']:
    site_id, _, r_id = subject_id.split('_')
    
    
    mmse = df[(df['SITEID'] == int(site_id))]
    if len(mmse) > 0:
        print len(mmse)

"""

for subject_id in dataset['subjects']:
    _, _, r_id = subject_id.split('_')
    for k in a:
        if k == int(r_id):
            print k
"""
"""
no_mmse_subjects = []
n_mmse = 0
n_no_mmse = 0
for subject_id in dataset['subjects']:
    mmse = df[df['Subject ID']==subject_id]['Functional Assessment Questionnaire Total Score'].values
    print mmse
    if np.all(np.isnan(mmse)):
        no_mmse_subjects.append(subject_id)
        n_no_mmse += 1
    else:
        n_mmse += 1

print no_mmse_subjects, n_no_mmse
print n_mmse

for subject_id in no_mmse_subjects:
    print df[df['Subject ID']==subject_id]['DX Group'].values[0]
"""