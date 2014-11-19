"""
A script that :
- computes a Masker from FDG PET (baseline uniform)
- cross-validates a linear SVM classifier
- computes a ROC curve and AUC
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
from sklearn import svm
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from nilearn.input_data import NiftiMasker
from joblib import delayed, Parallel


def shuffle_train_and_test(train, test):
        Xtrain, Xtest = x[train], x[test]
        Ytrain, Ytest = y[train], y[test]
        estim.fit(Xtrain,Ytrain)
        print pg_counter, counter
        score[counter, pg_counter] = estim.score(Xtest, Ytest)

def plot_shufflesplit(score, pairwise_groups):
    """Boxplot of the accuracies
    """
    plt.boxplot(score, labels=['/'.join(pg) for pg in pairwise_groups])
    plt.ylabel('Accuracy')
    plt.title('ADNI baseline accuracies (voxels)')
    plt.legend(loc="lower right")
    fname = 'boxplot_adni_baseline_voxels_norm'
    plt.savefig(os.path.join('figures', fname))
    

def plot_roc(cv_dict):
    """Plot roc curves for each pairwise groupe
    """

    for pg in cv_dict.keys():
        plt.plot(crossval[pg]['fpr'],crossval[pg]['tpr'],
                 label='{0} (auc = {1:0.2f})'
                                   ''.format(pg, crossval[pg]['auc']))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ADNI baseline ROC curves (voxels)')
        plt.legend(loc="lower right")

    fname = 'roc_adni_baseline_voxels_norm'
    plt.savefig(os.path.join('figures', fname))



BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))


if os.path.exists('features/features_voxels_norm.npy'):
    X = np.load('features/features_voxels_norm.npy')
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
    pet_masked = masker.transform_niimgs(pet_files, n_jobs=4)
    X = np.vstack(pet_masked)
    np.save('features/features_voxels_norm', X)


#AD/MCI, AD/Normal, MCI/LMCI, MCI/Normal
pairwise_groups = [['AD', 'Normal'], ['AD', 'EMCI'], ['AD', 'LMCI'],
                   ['EMCI', 'LMCI'], ['EMCI', 'Normal'], ['LMCI', 'Normal']]
nb_iter = 10
score = np.zeros((nb_iter, len(pairwise_groups)))
crossval = dict()
pg_counter = 0
for pg in pairwise_groups:
    gr1_idx = data[data.DX_Group == pg[0]].index.values
    gr2_idx = data[data.DX_Group == pg[1]].index.values
    x = X[np.concatenate((gr1_idx, gr2_idx))]
    y = np.ones(len(x))
    y[len(y) - len(gr2_idx):] = 0

    estim = svm.SVC(kernel='linear')
    sss = cross_validation.StratifiedShuffleSplit(y, n_iter=nb_iter, test_size=0.2)
    # 1000 runs with randoms 80% / 20% : StratifiedShuffleSplit
    counter = 0
    
    Parallel(n_jobs=6)(delayed(shuffle_train_and_test)((train, test))  for train, test in sss)

    """
    for train, test in sss:
        Xtrain, Xtest = x[train], x[test]
        Ytrain, Ytest = y[train], y[test]
        Yscore = estim.fit(Xtrain,Ytrain)
        print pg_counter, counter
        score[counter, pg_counter] = estim.score(Xtest, Ytest)
        counter += 1
    """

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
    crossval['/'.join(pg)] = {'fpr' : fpr,
                              'tpr' : tpr,
                              'thresholds' : thresholds,
                              'yproba' : yproba,
                              'auc' : a}
    pg_counter += 1


plot_roc(crossval)
plt.figure()
plot_shufflesplit(score, pairwise_groups)
plt.figure()