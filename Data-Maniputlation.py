import pandas as pd
import numpy as np
import os

def create_new_list_Stable_CN():
    df = pd.read_csv('/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv')

    # Group by 'Subject'
    df_grouped = df.groupby('Subject')

    stable_cn_data = []

    for subject_id, group in df_grouped:
        # Check if all 'Group' entries are 'CN' for this subject
        if (group['Group'] == 'CN').all():
            stable_cn_data.append(group)

    with open('Stable_CN_Subjects.txt', 'w') as f:
        for subject_id in stable_cn_subjects:
            f.write(f"{subject_id}\n")

    return result_df

def create_new_list_Stable_MCI():
    df = pd.read_csv('/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv')

    df_grouped = df.groupby('Subject')
    stable_mci_subjects = []

    for subject_id, group in df_grouped:
        if (group['Group'] == 'MCI').all():
            stable_mci_subjects.append(subject_id)

    with open('Stable_MCI_Subjects.txt', 'w') as f:
        for subject_id in stable_mci_subjects:
            f.write(f"{subject_id}\n")

    return stable_mci_subjects


def create_new_list_Stable_AD():
    df = pd.read_csv('/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv')

    df_grouped = df.groupby('Subject')
    stable_ad_subjects = []

    for subject_id, group in df_grouped:
        if (group['Group'] == 'AD').all():
            stable_ad_subjects.append(subject_id)

    with open('Stable_AD_Subjects.txt', 'w') as f:
        for subject_id in stable_ad_subjects:
            f.write(f"{subject_id}\n")

    return stable_ad_subjects


def create_list_CN_to_MCI():
    df = pd.read_csv('/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv')
    
    print("Columns:", df.columns)
    print("Unique groups:", df['Group'].unique())

    converters = []

    for subject_id, group in df.groupby('Subject'):
        group_sorted = group.sort_values('Visit_idx')
        group_labels = [g.strip().upper() for g in group_sorted['Group']]

        if 'CN' in group_labels and 'MCI' in group_labels:
            cn_index = group_labels.index('CN')
            try:
                mci_index = group_labels.index('MCI', cn_index + 1)
                if mci_index > cn_index:
                    converters.append(subject_id)
            except ValueError:
                continue  # No 'MCI' after 'CN'

    print(f"Found {len(converters)} converters from CN to MCI.")

    with open('CN_to_MCI_Subjects.txt', 'w') as f:
        for subject_id in converters:
            f.write(f"{subject_id}\n")

    return converters

def create_list_MCI_to_AD():
    df = pd.read_csv('/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv')

    converters = []

    for subject_id, group in df.groupby('Subject'):
        group_sorted = group.sort_values('Visit_idx')
        group_labels = group_sorted['Group'].tolist()

        # If the subject starts as MCI and later becomes AD
        if 'MCI' in group_labels and 'AD' in group_labels:
            first_group = group_labels[0]
            if first_group == 'MCI':
                if any(g == 'AD' for g in group_labels[1:]):
                    converters.append(subject_id)

    with open('MCI_to_AD_Subjects.txt', 'w') as f:
        for subject_id in converters:
            f.write(f"{subject_id}\n")

    return converters

create_list_CN_to_MCI()
