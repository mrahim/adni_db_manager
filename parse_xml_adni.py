"""
    Routines for parsing ADNI xml files
"""

import os, glob
import xml.etree.ElementTree as et

BASE_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data',
                        'pet_fdg_baseline_processed_ADNI')

xml_files = glob.glob(os.path.join(BASE_DIR, '*.xml'))

xml_all = et.fromstring('<subjects></subjects>')
for xml_file in xml_files:
    tree = et.parse(xml_file)
    subj = tree.find('project/subject')
    xml_all.append(subj)

req = "./subject/visit[visitIdentifier='ADNI1 Screening']"
req = "./subject/study/series/[modality='PET']"

req = '/'.join(['.', 'subject', 'study', 'series', 'seriesLevelMeta',
                'relatedImageDetail', 'originalRelatedImage',
                'protocolTerm', 'protocol[@term="Radiopharmaceutical"]'])

res = xml_all.findall(req)

for r in res:
    print r.text



"""
The elements of interest are :
    - ./subject/subjectIdentifier
    - ./subject/subjectInfo item="DX Group"
    - ./subject/visit[visitIdentifier='ADNI1 Screening']
    - ./subject/studyIdentifier
    - ./subject/study/series/modality
    - ./subject/study/series/seriesIdentifier
    - ./subject/study/series/dateAcquired
    - ./subject/study/series/seriesLevelMeta/derivedProduct/imageUID
    - ./subject/study/series/seriesLevelMeta/derivedProduct/processedDataLabel
    - ./subject/study/series/seriesLevelMeta/derivedProduct/relatedImage/imageUID
    - ./subject/study/series/seriesLevelMeta/
        relatedImageDetail/originalRelatedImage/imageUID
        relatedImageDetail/originalRelatedImage/description
        relatedImageDetail/originalRelatedImage/protocol term="Radiopharmaceutical"
        

"""