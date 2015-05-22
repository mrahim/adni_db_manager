# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 18:30:30 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob, shutil
import numpy as np
import pandas as pd


BASE_DIR = '/disk4t/mehdi/data/ADNI'
DST_DIR = '/disk4t/mehdi/data/ADNI_extracted'



## Clean DST_DIR
#if os.path.isdir(DST_DIR):
#    shutil.rmtree(DST_DIR)
#os.mkdir(DST_DIR)

subject_dir_list = sorted(glob.glob(os.path.join(BASE_DIR,
                                                 '[0-9][0-9][0-9]_S_*')))
subjects = []
for subj_dir in subject_dir_list:
    _, subject_id = os.path.split(subj_dir)
    fmri_dir = os.path.join(subj_dir, 'Resting_State_fMRI')
    if os.path.isdir(fmri_dir):
        fmri_seq = sorted(os.listdir(fmri_dir))
        subjects.append(subject_id)
        if len(fmri_seq) > 0:
            seq = os.path.join(fmri_dir, fmri_seq[0])
            img = os.path.join(seq, sorted(os.listdir(seq))[0])
            src = img
            dst = os.path.join(DST_DIR, subject_id, 'func')
#            # Clean func dst dir
#            if os.path.isdir(dst):
#                shutil.rmtree(dst)
#            shutil.copytree(src, dst)
        # handle anat img
        anat_dir = sorted(glob.glob(os.path.join(subj_dir, '*N3*')))
        if len(anat_dir) > 0:
            img = os.path.join(anat_dir[0],
                               sorted(os.listdir(anat_dir[0]))[0])
            img = os.path.join(img,
                               sorted(os.listdir(img))[0])
            src = img
            dst = os.path.join(DST_DIR, subject_id, 'anat')
#            # Clean anat dst dir
#            if os.path.isdir(dst):
#                shutil.rmtree(dst)
#            shutil.copytree(src, dst)
        else:
            print subject_id, 'has no anat'
        # handle pet img
        pet_dir = sorted(glob.glob(os.path.join(subj_dir,
                                                'Coreg*_Uniform_Resolution')))
        if len(pet_dir) > 0:
            seq = os.path.join(pet_dir[-1],
                               sorted(os.listdir(pet_dir[-1]))[0])
            img = os.path.join(seq, os.listdir(seq)[0])
            src = img
            dst = os.path.join(DST_DIR, subject_id, 'pet')
#            # Clean pet dst dir
#            if os.path.isdir(dst):
#                shutil.rmtree(dst)
#            shutil.copytree(src, dst)



subject_dir_list = sorted(glob.glob(os.path.join(DST_DIR,
                                                 '[0-9][0-9][0-9]_S_*')))
subjects = []
for subj_dir in subject_dir_list:
    _, subject_id = os.path.split(subj_dir)
    if len(os.listdir(os.path.join(subj_dir, 'func'))) == 0:
        print subject_id, 'func empty'
    if os.path.isdir(os.path.join(subj_dir, 'anat')):
        if len(os.listdir(os.path.join(subj_dir, 'anat'))) == 0:
            print subject_id, 'anat empty'
    else:
        print subject_id, 'anat empty'
    if os.path.isdir(os.path.join(subj_dir, 'pet')):
        if len(os.listdir(os.path.join(subj_dir, 'pet'))) == 0:
            print subject_id, 'pet empty'
    else:
        print subject_id, 'pet empty'


