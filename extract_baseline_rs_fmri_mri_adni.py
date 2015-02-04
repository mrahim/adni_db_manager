"""
Extract all baseline rs-fmri and mri images
"""

import os, glob, shutil
import pandas as pd

BASE_DIR = os.path.join('csv', 'fetch_results')

ADNI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'ADNI')

DST_BASE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                            'data', 'ADNI_baseline_rs_fmri_mri')

images = pd.read_csv(os.path.join(BASE_DIR, 'mri_fmri_rs_baseline.csv'))


for idx, row in images.iterrows():
    
    image_id = dict()
    xml_file = dict()
    
    # fMRI
    image_id['fmri'] = 'I' + str(row['Image_ID_x'])
    xml_file['fmri'] = glob.glob(os.path.join(ADNI_DIR,
                                 '*I' + str(row['Image_ID_x']) + '.xml'))

    # MRI    
    image_id['mri'] = 'I' + str(row['Image_ID_y'])
    xml_file['mri'] = glob.glob(os.path.join(ADNI_DIR,
                                '*I' + str(row['Image_ID_y']) + '.xml'))

    subject_id = row['Subject_ID']
    filename = dict()
    description = dict()
    folderpath = dict()
    for key in xml_file.keys():
        filename[key] = os.path.split(xml_file[key][0])[1].split('.')[0]
        description[key] = filename[key].split('_', 4)[4].rsplit('_', 2)[0]
        folderpath[key]= os.path.join(ADNI_DIR, subject_id, description[key])


    image_folder = {'fmri': 'func',
                    'mri': 'anat'}
    for key in ['fmri', 'mri']:
        for root, dirs, files in os.walk(folderpath[key]):
            if len(files) > 0:
                if image_id[key] in files[0]:
                    dst_path = os.path.join(DST_BASE_DIR, 's'+subject_id,
                                            image_folder[key])
                    if os.path.isdir(dst_path):
                        shutil.rmtree(dst_path)
                    shutil.copytree(root, dst_path)
                    shutil.copy(xml_file[key][0], DST_BASE_DIR)
    print subject_id

images.to_csv(os.path.join(DST_BASE_DIR, 'description_file.csv'))