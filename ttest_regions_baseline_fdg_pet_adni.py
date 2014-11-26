# -*- coding: utf-8 -*-
"""
T-test at region level of pairwise DX Groups.
From a normalised segmentation
Plot t-maps and p-maps on the regions.
@author: Mehdi
"""

# 1- Masking data
# 2- T-Testing

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import plot_stat_map
from scipy import stats

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'

MNI_TEMPLATE = os.path.join(BASE_DIR, 'wMNI152_T1_2mm_brain.nii')
SEG_TEMPLATE = os.path.join(BASE_DIR, 'wSegmentation.nii')
N_REGIONS = 83

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))
seg_img = nib.load(SEG_TEMPLATE)
seg_data = seg_img.get_data()

if os.path.exists('features/features_regions.npy'):
    x = np.load('features/features_regions.npy')
else:
    x = np.zeros((len(data), N_REGIONS))
    for idx, row in data.iterrows():
        pet_file = glob.glob(os.path.join(BASE_DIR,
                                          'I' + str(row.Image_ID), 'wI*.nii'))
        pet_img = nib.load(pet_file[0])
        pet_data = pet_img.get_data()
        for val in np.unique(seg_data):
            if val > 0:
                ind = (seg_data == val)
                x[idx, (val/256)-1] = np.mean(pet_data[ind])
    np.save('features/features_regions', x)

##############################################################################
# Inference
##############################################################################
groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]

for gr in groups:
    gr1_idx = data[data.DX_Group == gr[0]].index.values
    gr2_idx = data[data.DX_Group == gr[1]].index.values

    gr1_f = x[gr1_idx, :]
    gr2_f = x[gr2_idx, :]
    tval_region, pval_region = stats.ttest_ind(gr1_f, gr2_f)
    pval_region = - np.log10(pval_region)

    t_data = np.zeros(seg_data.shape)
    p_data = np.zeros(seg_data.shape)
    idx = 0
    for val in np.unique(seg_data):
        if val > 0:
            t_data[(seg_data == val)] = tval_region[idx]
            p_data[(seg_data == val)] = pval_region[idx]
            idx += 1
    t_img = nib.Nifti1Image(t_data, seg_img.get_affine())
    p_img = nib.Nifti1Image(p_data, seg_img.get_affine())
    
    t_nii_filename = '_'.join(['tmap', 'regions'])
    t_nii_filename += '_' + '_'.join(gr) + '.nii.gz'
    p_nii_filename = '_'.join(['pmap', 'regions'])
    p_nii_filename += '_' + '_'.join(gr) + '.nii.gz'
    
    t_img.to_filename(os.path.join('figures', 'nii', t_nii_filename))
    p_img.to_filename(os.path.join('figures', 'nii', p_nii_filename))

    plot_stat_map(t_img, bg_img=MNI_TEMPLATE, threshold='auto', black_bg=True,
                  title = '/'.join(gr))