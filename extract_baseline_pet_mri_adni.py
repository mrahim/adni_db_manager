"""
    Select images that have PET and segmented MRI
"""

import os, glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.join('csv', 'fetch_results')

ADNI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data', 'ADNI')

mri = pd.read_csv(os.path.join(BASE_DIR, 'mri_segmented_baseline.csv'))
pet = pd.read_csv(os.path.join(BASE_DIR, 'pet_fdg_uniform_baseline.csv'))

pet_mri = pd.merge(pet, mri, on='Subject_ID', how='inner')


# Create a directory which contains MRI and PET for all subjects

for image_id in np.concatenate((pet_mri['Image_ID_x'].values,
                                pet_mri['Image_ID_y'].values)):
    xml_list = glob.glob(os.path.join(ADNI_DIR, '*I' + str(image_id) + '.xml'))
    
    if len(xml_list)==0:
        continue
    
    xml_file = xml_list[0]
    filename = os.path.split(xml_file)[1].split('.')[0]
    subject_id = '_'.join(filename.split('_', 4)[1:4])
    description = filename.split('_', 4)[4].rsplit('_', 2)[0]
    sequence_id = filename.rsplit('_', 2)[1]
    folderpath = os.path.join(ADNI_DIR, subject_id, description)
    
    for d in os.listdir(folderpath):
        print sequence_id
        print d, os.listdir(os.path.join(folderpath, d))
        
    
    
    """
    for root, dirs, files in os.walk(folderpath):
        for d in dirs:
            #print d, image_id
            a = re.match('^[SI]' + str(image_id), d)
            print a, d, image_id
            if d == 'I' + str(image_id):
                image_folder = os.path.join(root, d)
                break
    #print image_folder
"""

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