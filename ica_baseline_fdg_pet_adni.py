"""
    CanICA on ADNI rs-fmri
"""
import os, glob
import numpy as np
import pandas as pd
from nilearn.plotting import plot_img
from nilearn.decomposition.canica import CanICA
from nilearn.input_data import MultiNiftiMasker
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'
CACHE_DIR = '/disk4t/mehdi/data/tmp'

data = pd.read_csv(os.path.join(BASE_DIR, 'description_file.csv'))
pet_files = []
for idx, row in data.iterrows():
    pet_file = glob.glob(os.path.join(BASE_DIR,
                                      'I' + str(row.Image_ID), 'wI*.nii'))
    if len(pet_file) > 0:
        pet_files.append(pet_file[0])
        """
        img = nib.load(pet_file[0])
        pet_img.append(img)
        """

n_sample = 140
idx = np.random.randint(len(pet_files), size=n_sample)
pet_files_sample = np.array(pet_files)[idx]

multi_masker = MultiNiftiMasker(mask_strategy='epi',
                                memory=CACHE_DIR,
                                n_jobs=1, memory_level=2)
multi_masker.fit(pet_files_sample)
plot_img(multi_masker.mask_img_)


n_components = 40
canica = CanICA(mask=multi_masker, n_components=n_components,
                smoothing_fwhm=6., memory=CACHE_DIR, memory_level=5,
                threshold=3., verbose=10, random_state=0)
canica.fit(pet_files_sample)


# Retrieve the independent components in brain space
components_img = canica.masker_.inverse_transform(canica.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('/disk4t/mehdi/data/tmp/canica_tep_140.nii.gz')

### Visualize the results #####################################################
# Show some interesting components

for i in range(n_components):
    plot_stat_map(nib.Nifti1Image(components_img.get_data()[..., i],
                                      components_img.get_affine()),
                  display_mode="z", title="IC %d"%i, cut_coords=1,
                  colorbar=False)
plt.show()