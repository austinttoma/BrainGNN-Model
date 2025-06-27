# Copyright (c) 2019 Mwiza Kunda
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import sys
import argparse
import numpy as np
import pandas as pd
import deepdish as dd

# === CONFIG ===
subject_file = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/subject_ID.txt'
label_file = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/TADPOLE_Simplified.csv'
fc_matrix_dir = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/FC_Matrix_Output_Parallel'
output_dir = os.path.join(fc_matrix_dir, 'raw_same')

# === HELPERS ===
def load_subject_ids(subject_file):
    with open(subject_file, 'r') as f:
        return [line.strip() for line in f]

def load_labels_per_visit(label_file):
    df = pd.read_csv(label_file)
    df['Group'] = df['Group'].replace({'CN': 0, 'MCI': 1, 'AD': 2})

    # Ensure required columns are present
    assert all(col in df.columns for col in ['Subject', 'Visit_idx', 'Group']), \
        "CSV must contain Subject, Visit_idx, and Group columns"

    # Sort by subject and visit (optional, but keeps order clean)
    df = df.sort_values(by=['Subject', 'Visit_idx'])

    # Build lookup dict: (subject_id, visit) → current label (Group)
    label_lookup = {
        (str(row['Subject']), int(row['Visit_idx'])): int(row['Group'])
        for _, row in df.iterrows()
    }

    return label_lookup

def load_fc_matrix(subject_id, run, kind='correlation'):
    fc_file = os.path.join(fc_matrix_dir, f"sub-{subject_id}_run-0{run}_fc.npy")
    if os.path.exists(fc_file):
        return np.load(fc_file)
    return None

# === MAIN ===
def main():
    parser = argparse.ArgumentParser(description='Prepare ADNI fMRI data for classification with next visit labels')
    parser.add_argument('--atlas', default='cc200', help='Atlas used: cc200, cc400, etc.')
    parser.add_argument('--nclass', default=3, type=int, help='Number of classes. CN=0, MCI=1, AD=2')
    parser.add_argument('--seed', default=123, type=int)
    args = parser.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    print("Parsers Clear")
    subject_ids = load_subject_ids(subject_file)
    label_lookup = load_labels_per_visit(label_file)
    print("SubjectID and Label Clear")
    for sid in subject_ids:
        for run in range(1, 10):
            fc = load_fc_matrix(sid, run)
            if fc is None:
                break  # No more runs for this subject

            label = label_lookup[key]
            save_path = os.path.join(output_dir, f"sub-{sid}_run-{run}.h5")
            dd.io.save(save_path, {'corr': fc, 'label': label})
            print(f"[✓] Saved: {save_path} with label={label}")

if __name__ == '__main__':
    main()
