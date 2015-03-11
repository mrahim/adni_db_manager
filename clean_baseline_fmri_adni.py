# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:58:04 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob, shutil

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri_mri'

subj_list = sorted(glob.glob(os.path.join(BASE_DIR, 's[0-9]*')))


for subj in subj_list:
    fmri_list = glob.glob(os.path.join(subj, 'func', '[zrw]*.nii'))
    for fmri in fmri_list:
        print fmri
        os.remove(fmri)