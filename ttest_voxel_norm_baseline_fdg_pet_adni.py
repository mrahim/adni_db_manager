# -*- coding: utf-8 -*-
"""
T-test on the voxels of pairwise DX groups.
Plot t-maps and p-maps on the voxels.
@author: Mehdi
"""

# 1- Masking data
# 2- T-Testing

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_roi, plot_stat_map, plot_img
from nilearn.mass_univariate import permuted_ols
from scipy import stats
from matplotlib import cm

def plot_mask(pet_files, pet_imgs):
    for pi, pf in zip(pet_imgs, pet_files):
        mask_path = os.path.join('figures', 'mask',
                                 pf.split('/')[-1].split('.')[0]) 
        plot_roi(masker.mask_img_, pi, output_file=mask_path,
                 title=pf.split('/')[-1].split('.')[0])

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
MNI_TEMPLATE = os.path.join(BASE_DIR, 'wMNI152_T1_2mm_brain.nii')

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))

pet_files = []
pet_img = []
for idx, row in data.iterrows():
    pet_file = glob.glob(os.path.join(BASE_DIR,
                                      'I' + str(row.Image_ID), 'wI*.nii'))
    if len(pet_file)>0:
        pet_files.append(pet_file[0])
        img = nib.load(pet_file[0])
        pet_img.append(img)

masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(opening=1))
masker.fit(pet_files)

plot_roi(masker.mask_img_, pet_file[0])

pet_masked = masker.transform_niimgs(pet_files, n_jobs=4)
pet_masked = np.vstack(pet_masked)
nb_vox = pet_masked.shape[1]

groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
          ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]

for gr in groups:
    gr1_idx = data[data.DX_Group == gr[0]].index.values
    gr2_idx = data[data.DX_Group == gr[1]].index.values
    
    gr1_f = pet_masked[gr1_idx, :]
    gr2_f = pet_masked[gr2_idx, :]
    
    t_masked, p_masked = stats.ttest_ind(gr1_f, gr2_f)
    p_masked = - np.log10(p_masked)


    gr_idx = np.hstack([gr1_idx, gr2_idx])        
    gr_f = pet_masked[gr_idx, :]
    gr_labels = np.vstack([np.hstack([[1]*len(gr1_idx), [0]*len(gr2_idx)]),
                           np.hstack([[0]*len(gr1_idx), [1]*len(gr2_idx)])]).T

    
    neg_log_pvals, t_scores, _ = permuted_ols(gr_labels, gr_f,
                                              n_perm=1000, n_jobs=6,
                                              model_intercept=True)

    tmap = masker.inverse_transform(t_masked)
    pmap = masker.inverse_transform(p_masked)
    
    tscore = masker.inverse_transform(t_scores[0])
    pscore = masker.inverse_transform(neg_log_pvals[0])

    t_path = os.path.join('figures',
                          'tmap_voxel_norm_'+gr[0]+'_'+gr[1]+'_baseline_adni')
    p_path = os.path.join('figures',
                          'pmap_voxel_norm_'+gr[0]+'_'+gr[1]+'_baseline_adni')
    
    plot_stat_map(tmap, tmap, output_file=t_path,
                  black_bg=True, title='/'.join(gr))
    """
    plot_stat_map(pmap, pmap, output_file=p_path,
                  black_bg=True, title='/'.join(gr))
    """
                  
    plot_img(pmap, bg_img=MNI_TEMPLATE, threshold=None,
             colorbar=True, cmap=cm.hot, vmin=0,
             output_file=p_path,
             black_bg=True, title='/'.join(gr))
                  
    tmap.to_filename(t_path+'.nii.gz')
    header = pmap.get_header()
    header['aux_file'] = 'hot'
    pmap.to_filename(p_path+'.nii.gz')
    
                  
    t_path = os.path.join('figures',
                          'tmap_perm_voxel_norm_'+gr[0]+'_'+gr[1]+'_baseline_adni')
    p_path = os.path.join('figures',
                          'pmap_perm_voxel_norm_'+gr[0]+'_'+gr[1]+'_baseline_adni')                  
                  
    plot_stat_map(tscore, tscore, output_file=t_path,
                  black_bg=True, title='/'.join(gr))
    """
    plot_stat_map(pscore, img, output_file=p_path,
                  black_bg=True, title='/'.join(gr))
    """
    
    plot_img(pmap, bg_img=MNI_TEMPLATE, threshold='auto',
             colorbar=True, cmap=cm.hot, vmin=0,
             output_file=p_path,
             black_bg=True, title='/'.join(gr))
                      

    tscore.to_filename(t_path+'.nii.gz')
    header = pscore.get_header()
    header['aux_file'] = 'Hot'
    pscore.to_filename(p_path+'.nii.gz')