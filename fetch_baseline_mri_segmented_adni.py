"""
    A script to fetch informations from ida search csv file:
    - Subject_ID
    - Image_ID
"""

import os
import pandas as pd

BASE_DIR = os.path.join('csv', 'ida_search')
DST_BASE_DIR = os.path.join('csv', 'fetch_results')
SEARCH_FILE = os.path.join(BASE_DIR, 'mri_segmented_maper_baseline.csv')

mri_segmented_baseline = pd.read_csv(SEARCH_FILE)
print len(mri_segmented_baseline)

mri_idx = [not 'masked' in d.lower() for d in mri_segmented_baseline['Description']]

mri_segmented_baseline = mri_segmented_baseline[mri_idx]

mri_segmented_baseline.to_csv(os.path.join(DST_BASE_DIR,
                                             'mri_segmented_baseline.csv'),
                                index=False)

subjects = mri_segmented_baseline['Subject_ID'].unique()

"""
# Write Subject_ID
f = open(os.path.join(DST_BASE_DIR,
                      'pet_fdg_uniform_baseline_id_subjects.txt'), 'w+')
f.write(','.join(subjects))
f.close()

# Write Image_ID
f = open(os.path.join(DST_BASE_DIR,
                      'pet_fdg_uniform_baseline_id_images.txt'), 'w+')
f.write(','.join(subjects))
f.close()
"""