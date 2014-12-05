"""
    Preprocessing ADNI baseline resting state fmri 
        - Alignement
        - Coreg MRI
        - Normalization MNI
        - Smoothing
"""
import os, glob
import numpy as np
import pandas as pd
from nilearn.plotting import plot_epi
from nilearn.image import mean_img

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_rs_fmri'
DST_DIR = '/disk4t/mehdi/data/tmp/quality_check'

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))
excluded_subjects = np.loadtxt(os.path.join(BASE_DIR, 'excluded_subjects'),
                         dtype='str')

for idx, row in data.iterrows():
    if not row.Subject_ID in excluded_subjects:
        fmri_file = glob.glob(os.path.join(BASE_DIR,
                                          'I' + str(row.Image_ID), '*.nii'))
    
        
    
