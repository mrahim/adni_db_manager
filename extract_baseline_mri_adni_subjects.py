"""
    extracting MRI images from PET subjects
"""
import os, glob, shutil
import pandas as pd

BASE_DIR = os.path.join('csv', 'fetch_results')

ADNI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'ADNI')

DST_BASE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                            'data', 'ADNI_baseline_fdg_pet_mri')

pet = pd.read_csv(os.path.join(BASE_DIR, 'pet_fdg_uniform_baseline.csv'))

mri = pd.read_csv(os.path.join(BASE_DIR, 'MRI.csv'))
mri = mri.drop_duplicates('Subject_ID')
merged = pd.merge(pet, mri, how='left', on='Subject_ID').drop_duplicates('Subject_ID')



for image_id in merged['Image_ID_y'].values:
    xml_list = glob.glob(os.path.join(ADNI_DIR, '*I' + str(image_id) + '.xml'))
    
    if len(xml_list)==0:
        print 'not found'
        continue
    
    print xml_list[0]
    break

