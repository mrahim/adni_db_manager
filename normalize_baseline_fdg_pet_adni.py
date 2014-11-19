# -*- coding: utf-8 -*-
"""
Normalization of adni fdg-pet to spm PET template
@author: Mehdi
"""

import os, glob
import pandas as pd
import nipype.interfaces.spm as spm


BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
SPM_DIR = '/i2bm/local/spm8'

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))

norm = spm.Normalize()
norm.inputs.template = os.path.join(SPM_DIR, 'templates', 'PET.nii')
norm.inputs.paths = SPM_DIR
for idx, row in data.iterrows():
    if idx > 562:
        pet_file = glob.glob(os.path.join(BASE_DIR,
                                          'I' + str(row.Image_ID), '*.nii'))
        if len(pet_file)>0:
            print '%s / %s' % (idx, len(data))
            norm.inputs.source = pet_file[0]
            norm.run()
