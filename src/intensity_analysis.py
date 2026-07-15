"""
intensity_analysis.py

Purpose:
--------
Analyze MRI intensity values and visualize their distribution.

Author: Belaynesh Kndie
"""

from pathlib import Path

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np


# ==========================================================
# MRI file
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

# ==========================================================
# Basic statistics
# ==========================================================

print("\n" + "=" * 50)
print("INTENSITY ANALYSIS")
print("=" * 50)

print(f"Shape: {data.shape}")
print(f"Min intensity: {data.min():.2f}")
print(f"Max intensity: {data.max():.2f}")
print(f"Mean intensity: {data.mean():.2f}")
print(f"Median intensity: {np.median(data):.2f}")
print(f"Standard deviation: {data.std():.2f}")

# ==========================================================
# Count background voxels
# ==========================================================

background_voxels = np.sum(data == 0)

total_voxels = data.size

print(f"\nTotal voxels: {total_voxels:,}")
print(f"Background voxels (0 intensity): {background_voxels:,}")
print(
    f"Background percentage: "
    f"{100 * background_voxels / total_voxels:.2f}%"
)

# ==========================================================
# Histogram
# ==========================================================

plt.figure(figsize=(10, 5))

plt.hist(
    data.flatten(),
    bins=100,
)

plt.title("MRI Intensity Distribution")
plt.xlabel("Intensity")
plt.ylabel("Number of Voxels")

plt.tight_layout()
plt.show()

# ==========================================================
# Histogram without background
# ==========================================================

brain_voxels = data[data > 0]

plt.figure(figsize=(10, 5))

plt.hist(
    brain_voxels,
    bins=100,
)

plt.title("MRI Intensity Distribution (Brain Only)")
plt.xlabel("Intensity")
plt.ylabel("Number of Voxels")

plt.tight_layout()
plt.show()