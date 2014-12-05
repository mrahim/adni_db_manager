"""
Extract all baseline rs-fmri images
"""

import os, glob, shutil
import pandas as pd

BASE_DIR = os.path.join('csv', 'fetch_results')

ADNI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'ADNI')

DST_BASE_DIR = os.path.join('/', 'disk4t', 'mehdi',
                            'data', 'ADNI_baseline_rs_fmri')

fmri = pd.read_csv(os.path.join(BASE_DIR, 'fmri_rs_baseline.csv'))


for image_id in fmri['Image_ID'].values:
    xml_list = glob.glob(os.path.join(ADNI_DIR, '*I' + str(image_id) + '.xml'))
    
    if len(xml_list)==0:
        print 'not found'
        continue
    
    xml_file = xml_list[0]
    filename = os.path.split(xml_file)[1].split('.')[0]
    subject_id = '_'.join(filename.split('_', 4)[1:4])
    description = filename.split('_', 4)[4].rsplit('_', 2)[0]
    sequence_id = filename.rsplit('_', 2)[1]
    folderpath = os.path.join(ADNI_DIR, subject_id, description)

    for root, dirs, files in os.walk(folderpath):
        for d in dirs:
            print d
            if d == str(sequence_id):
                image_folder = os.path.join(root, d)
                break
    print image_folder
    
    # Copy to a clean directory (nii, xml)
    dst_path = os.path.join(DST_BASE_DIR, 'I' + str(image_id))
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(image_folder, dst_path)
    shutil.copy(xml_file, DST_BASE_DIR)


fmri.to_csv(os.path.join(DST_BASE_DIR, 'description_file.csv'))