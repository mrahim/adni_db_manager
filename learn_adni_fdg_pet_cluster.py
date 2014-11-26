# -*- coding: utf-8 -*-
"""
Classification at cluster level of pairwise DX Groups.
Ward clustering of pet data.
@author: Mehdi
"""

import os, glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiMasker
from sklearn.feature_extraction import image
from sklearn.cluster import WardAgglomeration
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc

def plot_shufflesplit(score, pairwise_groups, N_CLUSTERS):
    bp = plt.boxplot(score, 0, '', 0)
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.grid(axis='x')
    plt.xlim([.4, 1.])
    plt.xlabel('Accuracy (%)', fontsize=18)
    plt.title('Shuffle Split Accuracies '+ str(N_CLUSTERS) +' clusters',
              fontsize=17)
    plt.yticks(range(1,7), ['AD/Normal', 'AD/EMCI', 'AD/LMCI', 'LMCI/Normal', 'LMCI/EMCI', 'EMCI/Normal'], fontsize=18)
    plt.xticks(np.linspace(0.4,1.0,7), np.arange(40,110,10), fontsize=18)
    plt.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['boxplot_adni_baseline_'+str(N_CLUSTERS)+'_clusters',
                          ext])
        plt.savefig(os.path.join('figures', fname), transparent=True)
        
        
def plot_shufflesplit_vert(score, pairwise_groups, N_CLUSTERS):
    """Boxplot of the accuracies
    """
    bp = plt.boxplot(score, labels=['/'.join(pg) for pg in pairwise_groups])
    for key in bp.keys():
        for box in bp[key]:
            box.set(linewidth=2)
    plt.grid(axis='y')
    plt.xticks([1, 1.9, 2.8, 3.8, 5, 6.3])
    plt.ylabel('Accuracy')
    plt.title('ADNI baseline accuracies ('+ str(N_CLUSTERS) +' clusters)')
    plt.legend(loc="lower right")
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['boxplot_adni_baseline_'+str(N_CLUSTERS)+'_clusters',
                          ext])
        plt.savefig(os.path.join('figures', fname))
        
        
        
def plot_roc(cv_dict, N_CLUSTERS):
    """Plot roc curves for each pairwise groupe
    """
   
    for pg in ['AD/Normal', 'AD/EMCI', 'AD/LMCI', 'LMCI/Normal', 'LMCI/EMCI', 'EMCI/Normal']:
        plt.plot(crossval[pg]['fpr'],crossval[pg]['tpr'],
                 linewidth=2,
                 label='{0} (auc = {1:0.2f})'
                                   ''.format(pg, crossval[pg]['auc']))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.title('K-Fold ROC curves ' + str(N_CLUSTERS) + ' clusters',
                  fontsize=18)
        plt.legend(loc="lower right")
        plt.tight_layout()
    
    for ext in ['png', 'pdf', 'svg']:
        fname = '.'.join(['roc_adni_baseline_'+str(N_CLUSTERS)+'_clusters',
                          ext])
        plt.savefig(os.path.join('figures', fname), transparent=True)

# 1- Ward (define nb_clusters)
# 2- mean intesnity cluster for each subject
# 3- t-test on clusters

###############################################################################


N_CLUSTERS_SET = [83]

#BASE_DIR = '/Volumes/CORSAIR/data/ADNI/ADNI_baseline_fdg_pet'
BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'

MNI_TEMPLATE = os.path.join(BASE_DIR, 'wMNI152_T1_2mm_brain.nii')

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))

"""
Test various n_clusters
"""
for N_CLUSTERS in N_CLUSTERS_SET:

    features_file = 'features_' + str(N_CLUSTERS) + '_clusters.npy'
    if os.path.exists(os.path.join('features', features_file)):
        X = np.load(os.path.join('features', features_file))
    else:
        
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
        
        ##############################################################################
        # Generate cluster matrix
        ##############################################################################
        
        X = np.zeros((len(data), N_CLUSTERS))
        for idx in np.arange(len(data)):
            for val in ward_labels_unique :
                ind = (ward_labels == val)
                X[idx, val] = np.mean(pet_data_masked[idx, ind])
            
        np.save(features_file, X)
    ##############################################################################
    # Classification
    ##############################################################################
    # Pairwise group comparison
    pairwise_groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
                       ['LMCI', 'Normal'], ['LMCI', 'EMCI'], ['EMCI', 'Normal']]
    nb_iter = 1000
    score = np.zeros((nb_iter, len(pairwise_groups)))
    crossval = dict()
    pg_counter = 0
    
    for gr in pairwise_groups:
        gr1_idx = data[data.DX_Group == gr[0]].index.values
        gr2_idx = data[data.DX_Group == gr[1]].index.values
        
        gr1_f = X[gr1_idx, :]
        gr2_f = X[gr2_idx, :]
        
        x = X[np.concatenate((gr1_idx, gr2_idx))]
        y = np.ones(len(x))
        y[len(y) - len(gr2_idx):] = 0
    
        estim = svm.SVC(kernel='linear')
        sss = cross_validation.StratifiedShuffleSplit(y,
                                                      n_iter=nb_iter,
                                                      test_size=0.2)
        # 1000 runs with randoms 80% / 20% : StratifiedShuffleSplit
        counter = 0
        for train, test in sss:
            Xtrain, Xtest = x[train], x[test]
            Ytrain, Ytest = y[train], y[test]
            Yscore = estim.fit(Xtrain,Ytrain)
            print pg_counter, counter
            score[counter, pg_counter] = estim.score(Xtest, Ytest)
            counter += 1
    
        # Cross-validation
        kf = cross_validation.StratifiedKFold(y,4)
        estim = svm.SVC(kernel='linear', probability=True)
        yproba = np.zeros((len(y), 2))
        
        for train, test in kf:
            xtrain, xtest = x[train], x[test]
            ytrain, ytest = y[train], y[test]
            yproba[test] = estim.fit(xtrain, ytrain).predict_proba(xtest)
            
        fpr, tpr, thresholds = roc_curve(1-y, yproba[:,0])
        a = auc(fpr,tpr)
        if a<.5:
            fpr, tpr, thresholds = roc_curve(y, yproba[:,0])
            a = auc(fpr,tpr)
        crossval['/'.join(gr)] = {'fpr' : fpr,
                                  'tpr' : tpr,
                                  'thresholds' : thresholds,
                                  'yproba' : yproba,
                                  'auc' : a}
        pg_counter += 1


    plot_roc(crossval, N_CLUSTERS)
    plt.figure()
    plot_shufflesplit(score, pairwise_groups, N_CLUSTERS)
    plt.figure()