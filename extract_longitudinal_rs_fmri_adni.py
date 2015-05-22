# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:14:32 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob, shutil
import numpy as np

ADNI_DIR = '/disk4t/mehdi/data/ADNI/'
DST_DIR = '/disk4t/mehdi/data/ADNI_extracted/fmri_longitudinal/'


subj_folders = sorted(glob.glob(os.path.join(ADNI_DIR, '[0-9][0-9][0-9]_S_*')))

# Collect rs-fmri files
func_list = []
for subj_folder in subj_folders:
    # glob rs-fmri folders
    rest_folders = []
    for reg in ['Ext*', 'Rest*']:
        rest_folders += sorted(glob.glob(os.path.join(subj_folder, reg)))
    
    # glob visits
    for rest_folder in rest_folders:
        visits = sorted(glob.glob(os.path.join(rest_folder, '[0-9]*')))
        # glob sequences
        for visit in visits:
            sequences = sorted(glob.glob(os.path.join(visit, 'S*')))
            # glob images
            for sequence in sequences:
                func_list += sorted(glob.glob(os.path.join(sequence, '*.nii')))

# Copy rs-fmri files
for func in func_list:
    data = func.split('/')
    subject_id = data[5]
    sequence_date = data[7]
    print data[5]
    subj_dir = os.path.join(DST_DIR, subject_id)
    func_dir = os.path.join(subj_dir, 'func')
    if not os.path.isdir(subj_dir):
        os.mkdir(subj_dir)
    if not os.path.isdir(func_dir):
        os.mkdir(func_dir)
    shutil.copy(func, func_dir)
