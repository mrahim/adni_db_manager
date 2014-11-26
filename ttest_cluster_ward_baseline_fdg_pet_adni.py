# -*- coding: utf-8 -*-
"""
T-test at cluster level of pairwise DX Groups.
Ward clustering of pet data.
Plot t-maps and p-maps on the voxels.
@author: Mehdi
"""

import os, glob
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_img, plot_stat_map
from nilearn.mass_univariate import permuted_ols
from sklearn.feature_extraction import image
from sklearn.cluster import WardAgglomeration
from scipy import stats

# 1- Ward (define nb_clusters)
# 2- mean intesnity cluster for each subject
# 3- t-test on clusters

###############################################################################


N_CLUSTERS_SET = [83]

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'

MNI_TEMPLATE = os.path.join(BASE_DIR, 'wMNI152_T1_2mm_brain.nii')

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))

pet_files = []
pet_img = []
for idx, row in data.iterrows():
    pet_file = glob.glob(os.path.join(BASE_DIR,
                                      'I' + str(row.Image_ID), 'wI*.nii'))
    if len(pet_file) > 0:
        pet_files.append(pet_file[0])
        img = nib.load(pet_file[0])
        pet_img.append(img)

masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(opening=1))
masker.fit(pet_files)

pet_data_masked = masker.transform_niimgs(pet_files, n_jobs=4)
pet_data_masked = np.vstack(pet_data_masked)

"""
Test various n_clusters
"""
for N_CLUSTERS in N_CLUSTERS_SET:

    ##############################################################################
    # Ward
    ##############################################################################
    
    mask = masker.mask_img_.get_data().astype(np.bool)
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                       n_z=shape[2], mask=mask)
    
    # Computing the ward for the first time, this is long...
    ward = WardAgglomeration(n_clusters=N_CLUSTERS, connectivity=connectivity,
                             memory='nilearn_cache')

    ward.fit(pet_data_masked)
    ward_labels_unique = np.unique(ward.labels_)
    ward_labels = ward.labels_
        
    ward_filename = '_'.join(['ward', str(N_CLUSTERS)])
    img_ward = masker.inverse_transform(ward.labels_)
    img_ward.to_filename(os.path.join('figures', 'nii', ward_filename))
    
    ##############################################################################
    # Generate cluster matrix
    ##############################################################################
    
    x = np.zeros((len(data), N_CLUSTERS))
    for idx in np.arange(len(data)):
        for val in ward_labels_unique :
            ind = (ward_labels == val)
            x[idx, val] = np.mean(pet_data_masked[idx, ind])
    
    
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
        tval_clustered, pval_clustered = stats.ttest_ind(gr1_f, gr2_f)
        pval_clustered = - np.log10(pval_clustered)
    
        gr_idx = np.hstack([gr1_idx, gr2_idx])
        gr_f = x[gr_idx, :]
        gr_labels = np.vstack([np.hstack([[1]*len(gr1_idx), [0]*len(gr2_idx)]),
                               np.hstack([[0]*len(gr1_idx), [1]*len(gr2_idx)])]).T
    
        p_clustered, t_clustered, _ = permuted_ols(gr_labels, gr_f,
                                                   n_perm=1000, n_jobs=4,
                                                   model_intercept=True)
    
        t_masked = np.zeros(pet_data_masked.shape[1])
        p_masked = np.zeros(pet_data_masked.shape[1])
        pval_masked = np.zeros(pet_data_masked.shape[1])
        for val in ward_labels_unique:
            t_masked[(ward_labels == val)] = t_clustered[0, val]
            p_masked[(ward_labels == val)] = p_clustered[0, val]
            pval_masked[(ward_labels == val)] = pval_clustered[val]
    
        tmap = masker.inverse_transform(t_masked)
        pmap = masker.inverse_transform(p_masked)
        pvalmap = masker.inverse_transform(pval_masked)
        header = pmap.get_header()
        header['aux_file'] = 'Hot'
        header = pvalmap.get_header()
        header['aux_file'] = 'Hot'
    
        t_nii_filename = '_'.join(['tmap', 'ward', str(N_CLUSTERS)])
        t_nii_filename += '_' + '_'.join(gr)    
        p_nii_filename = '_'.join(['pmap', 'ward', str(N_CLUSTERS)])
        p_nii_filename += '_' + '_'.join(gr)
        pval_nii_filename = '_'.join(['pvalmap', 'ward', str(N_CLUSTERS)])
        pval_nii_filename += '_' + '_'.join(gr)
        

        t_fig_filename = t_nii_filename
        p_fig_filename = p_nii_filename
        pval_fig_filename = pval_nii_filename
        t_nii_filename += '.nii.gz'
        p_nii_filename += '.nii.gz'
        pval_nii_filename += '.nii.gz'
    
        tmap.to_filename(os.path.join('figures', 'nii', t_nii_filename))
        pmap.to_filename(os.path.join('figures', 'nii', p_nii_filename))
        pvalmap.to_filename(os.path.join('figures', 'nii', pval_nii_filename))
    
        plot_stat_map(tmap, MNI_TEMPLATE, threshold=None, cmap=cm.bwr,
                      output_file=os.path.join('figures', 'tmap', t_fig_filename),
                      black_bg=True, title='/'.join(gr))
    
        plot_img(pmap, bg_img=MNI_TEMPLATE, threshold=None,
                 colorbar=True, cmap=cm.hot, vmin=0,
                 output_file=os.path.join('figures', 'pmap', p_fig_filename),
                 black_bg=True, title='/'.join(gr))
                 
        plot_img(pvalmap, bg_img=MNI_TEMPLATE, threshold=None,
                 colorbar=True, cmap=cm.hot, vmin=0,
                 output_file=os.path.join('figures', 'pmap', pval_fig_filename),
                 black_bg=True, title='/'.join(gr))