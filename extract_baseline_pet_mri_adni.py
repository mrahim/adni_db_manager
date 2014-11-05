"""
    Select images that have PET and segmented MRI
"""

import os, glob
import pandas as pd

BASE_DIR = os.path.join('csv', 'fetch_results')

ADNI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'ADNI')

mri = pd.read_csv(os.path.join(BASE_DIR, 'mri_segmented_baseline.csv'))
pet = pd.read_csv(os.path.join(BASE_DIR, 'pet_fdg_uniform_baseline.csv'))

pet_mri = pd.merge(pet, mri, on='Subject_ID', how='inner')

cfile_list = []

for subject_id in pet_mri['Subject_ID']:
    xml_file = glob.glob(os.path.join(ADNI_DIR,
                                      'ADNI_' + subject_id + '*.xml'))
    print xml_file
    break



"""
for image_id in pet_mri['Image_ID']:
    matching = ['I'+str(image_id)+'.xml' in fl for fl in file_df['filepath']]
    xml_file = file_df[matching]['filepath'].unique()
    if len(xml_file) > 0:
        filename = os.path.split(xml_file[0])[1].split('.')[0]
        subject_id = '_'.join(filename.split('_', 4)[1:4])
        description = filename.split('_', 4)[4].rsplit('_', 2)[0]
        folderpath = os.path.join(BASE_DIR, subject_id, description)
        sequence_id = filename.rsplit('_', 2)[1]
        for root, dirs, files in os.walk(folderpath):
            for d in dirs:
                if d == str(sequence_id):
                    image_folder = os.path.join(root, d)
                    break
        print image_folder
        # Copy to a clean directory (nii, xml)
        dst_path = os.path.join(DST_BASE_DIR, 'I' + str(image_id))
        if os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(image_folder, dst_path)
        shutil.copy(xml_file[0], DST_BASE_DIR)
        cfile_list.append(image_id)
        
"""