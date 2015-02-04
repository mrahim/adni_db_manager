"""
    A script to fetch informations from ida search csv file:
    - Subject_ID
    - Image_ID
"""

import os
import pandas as pd

BASE_DIR = os.path.join('csv', 'ida_search')
DST_BASE_DIR = os.path.join('csv', 'fetch_results')
FMRI_SEARCH_FILE = os.path.join(BASE_DIR, 'fmri_baseline.csv')
MRI_SEARCH_FILE = os.path.join(BASE_DIR, 'mri_baseline.csv')

fmri_baseline = pd.read_csv(FMRI_SEARCH_FILE)
mri_baseline = pd.read_csv(MRI_SEARCH_FILE)

rs_idx = ['resting' in d.lower() for d in fmri_baseline['Description']]

fmri_rs_baseline = fmri_baseline[rs_idx]
fmri_rs_baseline = fmri_baseline[rs_idx].drop_duplicates('Subject_ID')


mri_baseline = mri_baseline[mri_baseline.Description == 'MT1; N3m <- MPRAGE']
mri_baseline = mri_baseline.drop_duplicates('Subject_ID')


mri_fmri_rs_baseline = pd.merge(fmri_rs_baseline, mri_baseline,
                                on='Subject_ID', how='inner')

mri_fmri_rs_baseline.to_csv(os.path.join(DST_BASE_DIR,
                                         'mri_fmri_rs_baseline.csv'),
                            index=False)