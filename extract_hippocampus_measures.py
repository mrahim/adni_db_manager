# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 16:58:34 2015

@author: mr243268
"""

import numpy as np
import pandas as pd
from datetime import date
from fetch_data import set_group_indices


def rid_to_ptid(rid):
    """Returns patient id for a given rid
    """

    ptid = roster[roster.RID == rid]['PTID'].values
    if len(ptid) > 0:
        return ptid[0]
    else:
        return ''


def ptid_to_rid(ptid):
    """Returns roster id for a given patient id
    """

    rid = roster[roster.PTID == ptid]['RID'].values
    if len(rid) > 0:
        return rid[0]
    else:
        return ''


def get_dx(rid, exam=None):
    """Returns all diagnoses for a given
    rid and sid"""

    dates = dx[dx.RID == rid]['EXAMDATE'].values
    exam_dates = [date(int(d[:4]), int(d[5:7]), int(d[8:])) for d in dates]

    # DXCHANGE
    change = dx[dx.RID == rid]['DXCHANGE'].values
    curren = dx[dx.RID == rid]['DXCURREN'].values

    # change, curren have the same length
    dxchange = [int(np.nanmax([change[k], curren[k]]))
                for k in range(len(curren))]

    if exam is not None and len(exam_dates) > 0:
        exam_date, ind = find_closest_exam_date(exam, exam_dates)
        # TODO : return exam_date?
        return dxchange[ind]
    else:
        return -4


def find_closest_exam_date(acq_date, exam_dates):
    """Returns closest date and indice of the
    closest exam_date from acq_date"""

    diff = [abs(acq_date - e) for e in exam_dates]
    ind = np.argmin(diff)
    return exam_dates[ind], ind

dx_list = np.array(['None',
                    'Normal',
                    'MCI',
                    'AD',
                    'Normal->MCI',
                    'MCI->AD',
                    'Normal->AD',
                    'MCI->Normal',
                    'AD->MCI',
                    'AD->Normal'])

roster = pd.read_csv('Diagnosis/ROSTER.csv')
dx = pd.read_csv('Diagnosis/DXSUM_PDXCONV_ADNIALL.csv')
dob = pd.read_csv('Diagnosis/PTDEMOG.csv')
fs = pd.read_csv('Diagnosis/UCSFFSX51_05_20_15.csv')
fs = fs.drop_duplicates('RID')

# MRI data
rids = fs['RID'].values
ptids = map(rid_to_ptid, rids)
mri_ptid = ptids

# PET data
from fetch_data import fetch_adni_fdg_pet, fetch_adni_petmr
pet = fetch_adni_petmr()
pet_ptid = pet.subjects 

# fMRI data
from fetch_data import fetch_adni_rs_fmri_conn
fmri = fetch_adni_rs_fmri_conn('fmri_subjects_msdl_rois.npy')
fmri_ptid = fmri.subjects

# PETMR intersection
petmr_ptid = np.intersect1d(mri_ptid, pet_ptid)
petmr_rid = map(ptid_to_rid, petmr_ptid)

# PETMR + fMRI intersection
petfmr_ptid = np.intersect1d(petmr_ptid, fmri_ptid)
petfmr_rid = map(ptid_to_rid, petfmr_ptid)
rids = petfmr_rid

rids = petmr_rid

exams = map(lambda r: fs[fs.RID == r]['EXAMDATE'].values[0], rids)
exams = map(lambda e: date(int(e[:4]),
                           int(e[5:7]),
                           int(e[8:])), exams)

# Extract diagnosis
dxs = np.array(map(get_dx, rids, exams))
dx_group = dx_list[dxs]
dx_ = set_group_indices(dx_group)

# Extract hippocampus values
column_idx = np.arange(131, 147)
cols = ['ST' + str(c) + 'HS' for c in column_idx]
X = np.array(map(lambda r: fs[fs.RID == r][cols].values, rids))[:, 0, :]

# Extract clinical groups
idx_ = np.hstack((dx_['AD'], dx_['MCI']))
x = X[idx_, :]
y = np.array([1]*len(dx_['AD']) + [0]*len(dx_['MCI']))

# Remove nans
idx = np.array([~np.isnan(v).all() for v in x])
x = x[idx]
y = y[idx]

# Then, let's combine with some PET data

pet_idx = np.array(map(lambda p: np.where(np.array(pet.subjects) == p)[0][0],
                       petmr_ptid))
pet_imgs = np.array(pet.pet)
pet_imgs = pet_imgs[pet_idx]

from fetch_data import set_cache_base_dir, fetch_atlas, fetch_adni_masks
from nilearn.input_data import NiftiMapsMasker, NiftiLabelsMasker, NiftiMasker
import nibabel as nib

atlas_name = 'mayo'
CACHE_DIR = set_cache_base_dir()
atlas = fetch_atlas(atlas_name)
mask = fetch_adni_masks()['mask_petmr']

masker_global = NiftiMasker(mask_img=mask)
suv = masker_global.fit().transform_imgs(pet_imgs, n_jobs=20)
suv = np.array(suv)[:, 0, :]
x_pet_global = suv[idx_, :]
x_pet_global = x_pet_global[idx]

if len(nib.load(atlas).shape) == 3:
    masker = NiftiLabelsMasker(labels_img=atlas, mask_img=mask,
                               memory=CACHE_DIR, memory_level=2, verbose=0)
else:
    masker = NiftiMapsMasker(maps_img=atlas, mask_img=mask,
                             memory=CACHE_DIR, memory_level=2, verbose=0)
masker.fit()
suvr = masker.transform(pet_imgs)
x_pet = suvr[idx_, :]
x_pet = x_pet[idx]
x_petmr = np.hstack((x, x_pet))

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit

lr_mri = LogisticRegression()
lr_pet = LogisticRegression()
lr_pet_global = LogisticRegression()
lr_petmr = LogisticRegression()
lr_stack = LogisticRegression()

sss = StratifiedShuffleSplit(y, n_iter=100, test_size=.2, random_state=42)

accs = []
for train, test in sss:
    # MRI
    lr_mri.fit(x[train], y[train])
    # PET
    lr_pet.fit(x_pet[train], y[train])
    lr_pet_global.fit(x_pet_global[train], y[train])
    # PET + MRI
    lr_petmr.fit(x_petmr[train], y[train])
    """
    # PET + MRI stacking
    x_pm_train = np.vstack((lr_mri.decision_function(x[train]),
                            lr_pet.decision_function(x_pet[train]))).T
    x_pm_test = np.vstack((lr_mri.decision_function(x[test]),
                           lr_pet.decision_function(x_pet[test]))).T
    """
    # PET + MRI stacking
    x_pm_train = np.vstack((lr_mri.decision_function(x[train]),
                            lr_pet_global.decision_function(x_pet_global[train]))).T
    x_pm_test = np.vstack((lr_mri.decision_function(x[test]),
                           lr_pet_global.decision_function(x_pet_global[test]))).T
    lr_stack.fit(x_pm_train, y[train])
    scores = [lr_mri.score(x[test], y[test]),
              lr_pet.score(x_pet[test], y[test]),
              lr_pet_global.score(x_pet_global[test], y[test]),
              lr_petmr.score(x_petmr[test], y[test]),
              lr_stack.score(x_pm_test, y[test])]
    accs.append(scores)

print np.mean(accs, axis=0)
