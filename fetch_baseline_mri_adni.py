"""
    A script to fetch informations from ida search csv file:
    - Subject_ID
    - Image_ID
"""

import os
import pandas as pd


FMRI_DIR = os.path.join('/', 'disk4t', 'mehdi',
                        'data', 'ADNI_baseline_rs_fmri')
BASE_DIR = os.path.join('csv', 'ida_search')
DST_BASE_DIR = os.path.join('csv', 'fetch_results')
SEARCH_FILE = os.path.join(BASE_DIR, 'mri_baseline.csv')

mri_baseline = pd.read_csv(SEARCH_FILE)

n3_idx = ['n3' in d.lower() for d in mri_baseline['Description']]

mri_baseline = mri_baseline[mri_baseline.Description == 'MT1; N3m <- MPRAGE']
mri_baseline = mri_baseline.drop_duplicates('Subject_ID')
fmri_baseline = pd.read_csv(os.path.join(FMRI_DIR, 'description_file.csv'))

mri_baseline.to_csv(os.path.join(DST_BASE_DIR,
                                    'fmri_rs_baseline.csv'),
                                    index=False)

subjects = mri_baseline['Subject_ID'].values
images = mri_baseline['Image_ID'].values


# Write Subject_ID
f = open(os.path.join(DST_BASE_DIR,
                      'mri_baseline_id_subjects.txt'), 'w+')
f.write(','.join(subjects))
f.close()

# Write Image_ID
f = open(os.path.join(DST_BASE_DIR,
                      'mri_baseline_id_images.txt'), 'w+')
f.write(','.join(subjects))
f.close()