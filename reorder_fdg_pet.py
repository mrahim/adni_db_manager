# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:24:08 2015

@author: mehdi.rahim@cea.fr
"""

import os, glob, shutil
import numpy as np
import pandas as pd

src_dir = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
dst_dir = '/disk4t/mehdi/data/ADNI_pet'

d = pd.read_csv(os.path.join(src_dir, 'description_file.csv'))

for idx, row in d.iterrows():

    subj_dir = os.path.join(src_dir, 'I' + str(row['Image_ID']))
    if os.path.isdir(subj_dir):
        subj_dir_dst = os.path.join(dst_dir,
                                    's' + str(row['Subject_ID']),
                                    'pet')
        print subj_dir_dst

        shutil.copytree(subj_dir, subj_dir_dst)
print d.keys()