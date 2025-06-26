# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
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

'''
This script mainly refers to https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/fetch_data.py
'''
def main():
        import os
        import numpy as np
        from nilearn import datasets, input_data, connectome
        from nilearn.maskers import NiftiLabelsMasker  # instead of input_data.NiftiLabelsMasker
        
        # === CONFIG ===
        fmri_dir = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/fmriprep_output'
        subject_file = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/subject_ID.txt'
        output_dir = '/media/volume/ADNI-Data/git/BrainGNN-Model/data/FC_Matrix_Output'
        atlas_name = 'cc200'
        correlation_type = 'correlation'

        # === ATLAS LOADING ===
        atlas_data = datasets.fetch_atlas_aal()
        atlas_img = atlas_data.maps
        labels = atlas_data.labels
        masker = input_data.NiftiLabelsMasker(labels_img=atlas_img,
                                            standardize=True,
                                            detrend=True,
                                            memory='nilearn_cache',
                                            verbose=0)

        # === OUTPUT DIRECTORY ===
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # === SUBJECT LOOP ===
        with open(subject_file, 'r') as f:
            subjects = [line.strip() for line in f]

        for sid in subjects:
            current_run = 1
            while True:
                nifti_path = os.path.join(
                    fmri_dir,
                    f"sub-{sid}",
                    "func",
                    f"sub-{sid}_task-rest_run-0{current_run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                )

                if not os.path.exists(nifti_path):
                    # If this run does not exist, break and move to next subject
                    if current_run == 1:
                        print(f"[!] No runs found for subject {sid}")
                    break  # exit while loop, move to next subject

                try:
                    print(f"[+] Processing subject {sid}, run {current_run} ...")
                    timeseries = masker.fit_transform(nifti_path)

                    # Compute connectivity matrix
                    conn = connectome.ConnectivityMeasure(kind=correlation_type)
                    matrix = conn.fit_transform([timeseries])[0]

                    # Save matrix
                    save_path = os.path.join(output_dir, f"sub-{sid}_run-0{current_run}_fc.npy")
                    np.save(save_path, matrix)
                    print(f"[✓] Saved to {save_path}")

                except Exception as e:
                    print(f"[!] Error processing sub-{sid} run-{current_run}: {e}")

                current_run += 1  # move to next run


if __name__ == '__main__':
    main()
