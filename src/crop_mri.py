"""
crop_mri.py

Purpose:
--------
Remove empty background from a brain MRI by finding the
smallest bounding box containing all non-zero voxels.

Author: Belaynesh Kndie
"""

from pathlib import Path

import nibabel as nib
import numpy as np


# ==========================================================
# MRI path
# ==========================================================

MRI_PATH = Path(
    "data/raw/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr"
)


# ==========================================================
# Load MRI
# ==========================================================

image = nib.load(str(MRI_PATH))
data = image.get_fdata()

# Remove singleton dimension
data = np.squeeze(data)

print("\nOriginal shape:")
print(data.shape)


# ==========================================================
# Create brain mask
# ==========================================================

mask = data > 0

# Count brain voxels
brain_voxels = np.sum(mask)

print(f"\nBrain voxels: {brain_voxels:,}")


# ==========================================================
# Find bounding box
# ==========================================================

x, y, z = np.where(mask)

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

print("\nBounding box:")
print(f"X: {x_min} -> {x_max}")
print(f"Y: {y_min} -> {y_max}")
print(f"Z: {z_min} -> {z_max}")


# ==========================================================
# Crop MRI
# ==========================================================

cropped = data[
    x_min:x_max + 1,
    y_min:y_max + 1,
    z_min:z_max + 1
]

print("\nCropped shape:")
print(cropped.shape)


# ==========================================================
# Space reduction
# ==========================================================

original_voxels = data.size
cropped_voxels = cropped.size

reduction = (
    (original_voxels - cropped_voxels)
    / original_voxels
) * 100

print(
    f"\nSpace reduction: "
    f"{reduction:.2f}%"
)


# ==========================================================
# Save cropped MRI
# ==========================================================

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "cropped_mri.nii.gz"

cropped_img = nib.Nifti1Image(
    cropped,
    affine=np.eye(4)
)

nib.save(cropped_img, str(output_path))

print(f"\nSaved cropped MRI:")
print(output_path)