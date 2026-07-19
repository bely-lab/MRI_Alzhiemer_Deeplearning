"""
preprocess_mri.py

Steps:
1. Load MRI
2. Remove singleton dimension
3. Crop background
4. Normalize intensities (Z-score)
5. Save processed MRI
"""

from pathlib import Path

import nibabel as nib
import numpy as np


# ==========================================================
# Paths
# ==========================================================

INPUT_PATH = Path(
    "data/raw/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr"
)

OUTPUT_PATH = Path(
    "processed/OAS1_0001_preprocessed.nii.gz"
)


# ==========================================================
# Helper Functions
# ==========================================================

def load_mri(path):
    """Load MRI and return image + data."""
    image = nib.load(str(path))
    data = image.get_fdata()

    # Remove singleton channel dimension
    data = np.squeeze(data)

    return image, data


def crop_background(data):
    """Crop MRI using non-zero voxels."""

    mask = data > 0

    x, y, z = np.where(mask)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    cropped = data[
        x_min:x_max + 1,
        y_min:y_max + 1,
        z_min:z_max + 1
    ]

    return cropped


def zscore_normalize(data):
    """
    Normalize only non-zero voxels.

    Background remains zero.
    """

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
    """Save MRI as NIfTI."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image = nib.Nifti1Image(
        data,
        affine=np.eye(4)
    )

    nib.save(image, str(output_path))


# ==========================================================
# Main Pipeline
# ==========================================================

print("\nLoading MRI...")
image, data = load_mri(INPUT_PATH)

print(f"Original shape: {data.shape}")

print("\nCropping background...")
cropped = crop_background(data)

print(f"Cropped shape: {cropped.shape}")

print("\nNormalizing intensities...")
normalized = zscore_normalize(cropped)

print(
    f"Normalized mean: "
    f"{normalized[normalized > 0].mean():.4f}"
)

print(
    f"Normalized std: "
    f"{normalized[normalized > 0].std():.4f}"
)

print("\nSaving MRI...")
save_mri(normalized, OUTPUT_PATH)

print(f"Saved: {OUTPUT_PATH}")

print("\nPreprocessing complete.")