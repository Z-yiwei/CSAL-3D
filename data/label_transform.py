import os
import numpy as np
import nibabel as nib
from glob import glob


def merge_whole_tumor(label):
    return (label > 0).astype(np.uint8)


def process_labels(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all .nii.gz files under the input directory
    label_paths = glob(os.path.join(input_dir, "*.nii.gz"))

    for label_path in label_paths:

        label_nii = nib.load(label_path)
        label_data = label_nii.get_fdata()

        whole_tumor_label = merge_whole_tumor(label_data)

        whole_tumor_nii = nib.Nifti1Image(whole_tumor_label, label_nii.affine, label_nii.header)

        output_path = os.path.join(output_dir, os.path.basename(label_path))
        nib.save(whole_tumor_nii, output_path)

        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = "Task01_BrainTumour/labelsTr"
    output_dir = "Task01_BrainTumour/labelsTr_merge"

    process_labels(input_dir, output_dir)
