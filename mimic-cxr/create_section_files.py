# This script extracts the conclusion section from MIMIC-CXR reports
# It outputs them into individual files with at most 10,000 reports.
import json
import sys
import os
import argparse
import csv
from pathlib import Path

from tqdm import tqdm

# local folder import
import section_parser as sp
from local_config import PATH_TO_MIMIC_CXR

parser = argparse.ArgumentParser()
parser.add_argument('--reports_path',
                    default=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-reports/files",
                    help=('Path to file with radiology reports,'
                          ' e.g. /data/mimic-cxr/files'))
parser.add_argument('--mimic_cxr_jpg_path',
                    default=f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/files",
                    help=('Path to file with radiology reports,'
                          ' e.g. /data/mimic-cxr/files'))
parser.add_argument('--output_path',
                    default='reports_processed',
                    help='Path to output CSV files.')


def list_rindex(l, s):
    """Helper function: *last* matching element in a list"""
    return len(l) - l[-1::-1].index(s) - 1


def main(args):
    args = parser.parse_args(args)

    reports_path = Path(args.reports_path)
    mimic_cxr_jpg_path = Path(args.mimic_cxr_jpg_path)
    output_path = Path(args.output_path)

    if not output_path.exists():
        output_path.mkdir()

    # not all reports can be automatically sectioned
    # we load in some dictionaries which have manually determined sections
    custom_section_names, custom_indices = sp.custom_mimic_cxr_rules()

    # get all higher up folders (p00, p01, etc)
    p_grp_folders = os.listdir(reports_path)
    p_grp_folders = [p for p in p_grp_folders
                     if p.startswith('p') and len(p) == 3]
    p_grp_folders.sort()

    # study_sections will have an element for each study
    # this element will be a list, each element having text for a specific section
    study_sections = []
    for p_grp in p_grp_folders:
        # get patient folders, usually around ~6k per group folder
        cxr_path = reports_path / p_grp
        p_folders = os.listdir(cxr_path)
        p_folders = [p for p in p_folders if p.startswith('p')]
        p_folders.sort()

        # For each patient in this grouping folder
        print(p_grp)
        for p in tqdm(p_folders):
            patient_path = cxr_path / p

            # get the filename for all their free-text reports
            studies = os.listdir(patient_path)
            studies = [s for s in studies if s.startswith('s')]

            for s in studies:

                img_path = mimic_cxr_jpg_path / p_grp / p / s.replace('.txt', '')
                corr_dicom_ids = os.listdir(img_path)
                corr_dicom_ids = [d.replace('.jpg', '') for d in corr_dicom_ids if d.endswith('.jpg')]
                # load in the free-text report
                with open(patient_path / s, 'r') as fp:
                    text = ''.join(fp.readlines())

                # get study string name without the txt extension
                s_stem = s[0:-4]

                # split text into sections
                sections, section_names, section_idx = sp.section_text(
                    text
                )

                study_sectioned = [s_stem]
                for sn in ('impression', 'findings',
                           'last_paragraph', 'comparison'):
                    if sn in section_names:
                        idx = list_rindex(section_names, sn)
                        study_sectioned.append(sections[idx].strip())
                    else:
                        study_sectioned.append(None)
                # append once per dicom_id
                for dicom_id in corr_dicom_ids:
                    study_sectioned_copy = study_sectioned.copy()
                    study_sectioned_copy.append(dicom_id)
                    study_sectioned_copy.append(f"{dicom_id}.jpg")
                    study_sectioned_copy.append(Path("files")/p_grp / p / s.replace('.txt', ''))
                    study_sectioned_copy.append(f'{s_stem}.txt')
                    study_sections.append(study_sectioned_copy)

    # write out a single CSV with the sections
    with open(output_path / 'mimic_cxr_sectioned.csv', 'w') as fp:
        csvwriter = csv.writer(fp)
        # write header
        csvwriter.writerow(['impression', 'findings', 'last_paragraph', 'comparison', 'dicom_id', 'Img_Filename', 'Img_Folder', 'Note_file'])
        for row in study_sections:
            csvwriter.writerow(row)


if __name__ == '__main__':
   main(sys.argv[1:])
