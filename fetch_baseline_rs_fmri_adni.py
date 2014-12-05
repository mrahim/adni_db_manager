"""
    A script to fetch informations from ida search csv file:
    - Subject_ID
    - Image_ID
"""

import os
import pandas as pd

BASE_DIR = os.path.join('csv', 'ida_search')
DST_BASE_DIR = os.path.join('csv', 'fetch_results')
SEARCH_FILE = os.path.join(BASE_DIR, 'fmri_baseline.csv')

fmri_baseline = pd.read_csv(SEARCH_FILE)

rs_idx = ['resting' in d.lower() for d in fmri_baseline['Description']]

fmri_rs_baseline = fmri_baseline[rs_idx]

fmri_rs_baseline = fmri_baseline[rs_idx].drop_duplicates('Subject_ID')


fmri_rs_baseline.to_csv(os.path.join(DST_BASE_DIR,
                                             'fmri_rs_baseline.csv'),
                                index=False)

subjects = fmri_rs_baseline['Subject_ID'].values
images = fmri_rs_baseline['Image_ID'].values


# Write Subject_ID
f = open(os.path.join(DST_BASE_DIR,
                      'fmri_rs_baseline_id_subjects.txt'), 'w+')
f.write(','.join(subjects))
f.close()

# Write Image_ID
f = open(os.path.join(DST_BASE_DIR,
                      'fmri_rs_baseline_id_images.txt'), 'w+')
f.write(','.join(subjects))
f.close()