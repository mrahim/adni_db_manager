"""
    A script to fetch informations from ida search csv file:
    - Subject_ID
    - Image_ID
"""

import os
import pandas as pd

BASE_DIR = os.path.join('csv', 'ida_search')
DST_BASE_DIR = os.path.join('csv', 'fetch_results')
SEARCH_FILE = os.path.join(BASE_DIR, 'pet_f_uniform_baseline.csv')


pet_uniform_baseline = pd.read_csv(SEARCH_FILE)

fdg_idx = ['fdg' in d.lower() for d in pet_uniform_baseline['Description']]

pet_fdg_uniform_baseline = pet_uniform_baseline[fdg_idx]

pet_fdg_uniform_baseline.to_csv(os.path.join(DST_BASE_DIR,
                                             'pet_fdg_uniform_baseline.csv'),
                                index=False)

subjects = pet_fdg_uniform_baseline['Subject_ID'].values
images = pet_fdg_uniform_baseline['Image_ID'].values


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
