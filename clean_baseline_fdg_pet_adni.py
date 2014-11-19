# -*- coding: utf-8 -*-
"""
Remove unused dcm files of adni fdg-pet
@author: Mehdi
"""

import os, glob

#BASE_DIR = '/disk4t/mehdi/data/pet_fdg_baseline_processed_ADNI'
BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'


for root, dirs, files in os.walk(BASE_DIR):
    file_list = glob.glob(root + "/ADNI*.dcm")     # Get dcm files
    if len(file_list) > 0:
        print root
        cmd = 'rm ' + ' '.join(file_list)
        os.system(cmd)