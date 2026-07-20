"""
process_dataset.py

Pipeline:
1. Load MRI
2. Remove singleton dimension
3. Crop background
4. Z-score normalization
5. Save processed MRI

Author: Belaynesh Kndie
"""

from pathlib import Path
import time

import nibabel as nib
import numpy as np


# ==========================================================
# Paths
# ==========================================================

DATASET_DIR = Path("data/raw/OASIS")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Helper Functions
# ==========================================================

def load_mri(path):
    image = nib.load(str(path))
    data = image.get_fdata()

    return np.squeeze(data)


def crop_background(data):
    mask = data > 0

    x, y, z = np.where(mask)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    return data[
        x_min:x_max + 1,
        y_min:y_max + 1,
        z_min:z_max + 1
    ]


def zscore_normalize(data):
    normalized = data.copy()

    mask = normalized > 0

    brain_voxels = normalized[mask]

    mean = brain_voxels.mean()
    std = brain_voxels.std()

    normalized[mask] = (
        brain_voxels - mean
    ) / std

    return normalized


def save_mri(data, output_path):
    image = nib.Nifti1Image(
        data,
        affine=np.eye(4)
    )

    nib.save(image, str(output_path))


# ==========================================================
# Main
# ==========================================================

start_time = time.time()

processed_count = 0
failed_count = 0

print("\nStarting dataset preprocessing...\n")

for subject_dir in sorted(DATASET_DIR.glob("OAS1_*")):

    try:

        hdr_files = list(subject_dir.glob("RAW/*.hdr"))

        if not hdr_files:
            print(f"Skipping {subject_dir.name} (no MRI found)")
            continue

        mri_path = hdr_files[0]

        print(f"Processing {subject_dir.name}")

        data = load_mri(mri_path)

        cropped = crop_background(data)

        normalized = zscore_normalize(cropped)

        output_file = (
            OUTPUT_DIR /
            f"{subject_dir.name}.nii.gz"
        )

        save_mri(
            normalized,
            output_file
        )

        processed_count += 1

    except Exception as e:

        failed_count += 1

        print(
            f"Failed {subject_dir.name}: {e}"
        )

end_time = time.time()

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)

print(f"Processed: {processed_count}")
print(f"Failed: {failed_count}")

print(
    f"Time: "
    f"{end_time - start_time:.2f} seconds"
)