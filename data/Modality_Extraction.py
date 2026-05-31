import os
import numpy as np
import nibabel as nib
from glob import glob

def extract_flair_modality(image):
    """
    Extract the FLAIR modality.
    Args:
        image (numpy array): shape (240, 240, 155, 4).
    Returns:
        numpy array: FLAIR volume of shape (240, 240, 155).
    """
    return image[..., 0]

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Collect all .nii.gz files under the input directory
    image_paths = glob(os.path.join(input_dir, "*.nii.gz"))

    for image_path in image_paths:
        # Load the image file
        image_nii = nib.load(image_path)
        image_data = image_nii.get_fdata()

        # Extract the FLAIR modality
        flair_image = extract_flair_modality(image_data)

        # Build a new NIfTI image object
        flair_nii = nib.Nifti1Image(flair_image, image_nii.affine, image_nii.header)

        # Save to the output directory
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        nib.save(flair_nii, output_path)

        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    input_dir = "./Task01_BrainTumour/imagesTr"
    output_dir = "./Task01_BrainTumour/imagesTr_flair"

    process_images(input_dir, output_dir)
