# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 10:43:32 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob
from fetch_data import set_data_base_dir

pet_folder = set_data_base_dir('ADNI_baseline_fdg_pet')
adni_folder = set_data_base_dir('ADNI')

subj_list = sorted(glob.glob(os.path.join(pet_folder, 's[0-9]*')))

for subj in subj_list:
    _, subj_id = os.path.split(subj)
    folders = sorted(glob.glob(os.path.join(adni_folder, subj_id[1:], '*N3*')))
#    if len(folders) == 0:
#        folders = sorted(glob.glob(os.path.join(adni_folder, subj_id[1:], '*MPR*')))
    if len(folders) == 0:
        folders = sorted(glob.glob(os.path.join(adni_folder, subj_id[1:], '*')))
        print subj_id[1:]
