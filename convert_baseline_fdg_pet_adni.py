# -*- coding: utf-8 -*-
"""
A script that converts dcm files in a directory to nii format
by using xmedcon
"""

import os, glob

def sort_dcm_files(dcm_files):
    '''
    returns a sorted dcm file list according to their specific number
    '''
    idx = []
    for f in dcm_files:
        idx.append(int(f.split('_')[-3]))
    dcm_dict = dict(zip(idx, dcm_files))
    return dcm_dict.values()
#
#
#BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet'

BASE_DIR = '/disk4t/mehdi/data/ADNI_baseline_fdg_pet_mri'
wdir = os.getcwd()
subject_dirs = glob.glob(os.path.join(BASE_DIR, 's*'))

   
for subject_dir in subject_dirs:
    file_list = glob.glob(os.path.join(subject_dir, 'pet', 'ADNI*.dcm'))# Get dcm files
    if len(file_list) > 0:
        del_files = glob.glob(os.path.join(subject_dir, 'pet', 'm00*'))
        for f in del_files:
            os.remove(f) # Delete possible old conversions
        file_list = sort_dcm_files(sorted(file_list))
        dcm_files = ' '.join(file_list) # Construct a string of dcm files
        _, image_id = os.path.split(file_list[0])
        image_id, _ = os.path.splitext(image_id)
        image_id = image_id.rsplit('_')[-1]
        medcon_cmd = ' '.join(['medcon', '-n', '-stack3d', '-qc',
                               '-fh', '-fv',
                               '-c', 'dicom', 'nifti', '-w',
                               '-o', image_id, '-f', dcm_files])                     
        os.chdir(os.path.join(subject_dir, 'pet'))
        os.system(medcon_cmd)   # Convert by using the (x)medcon command        
        print image_id


"""
wdir = os.getcwd()


for root, dirs, files in os.walk(BASE_DIR):
    file_list = glob.glob(root + "/ADNI*.dcm")     # Get dcm files
    if len(file_list) > 0:
        del_files = glob.glob(root + "/m00*")
        for f in del_files:
            os.system(' '.join(['rm', f]))  # Delete possible old conversions
                
        image_id = root.rsplit('/')[-1]
        file_list.sort()    # Sort dcm files
        file_list = sort_dcm_files(file_list)
        dcm_files = ' '.join(file_list) # Construct a string of dcm files
        medcon_cmd = ' '.join(['medcon', '-n', '-qs', '-stack3d',
                               '-fh', '-fv',
                               '-c', 'dicom', 'nifti', '-w',
                               '-o', image_id, '-f', dcm_files])
        os.chdir(root)
        os.system(medcon_cmd)   # Convert by using the (x)medcon command
        print image_id, len(file_list)
        
        os.chdir(wdir)
        f = open(os.path.join('reports', image_id + '_conv.txt'), 'w+')
        f.write('\n'.join(file_list))
        f.close()
        
"""