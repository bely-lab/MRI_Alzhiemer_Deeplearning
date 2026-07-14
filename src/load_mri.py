
# ==========================
# Import libraries
# ==========================

import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# ==========================
# Path to one MRI scan
# ==========================

MRI_PATH = Path(
    "data/raw/disc1/OAS1_0001_MR1/RAW/OAS1_0001_MR1_mpr-1_anon.hdr"
)

# ==========================
# Load the MRI
# ==========================

# nib.load() automatically reads both the .hdr and .img files.
image = nib.load(MRI_PATH)

# Convert the MRI into a NumPy array.
# The MRI is stored as a 3D matrix (volume).
data = image.get_fdata()

# ==========================
# Print image information
# ==========================

print("=" * 50)
print("MRI INFORMATION")
print("=" * 50)

print(f"Image shape: {data.shape}")
print(f"Data type: {data.dtype}")

# Voxel size tells us the physical size of each voxel.
print(f"Voxel size (mm): {image.header.get_zooms()}")

print(f"Minimum intensity: {data.min():.2f}")
print(f"Maximum intensity: {data.max():.2f}")

print("=" * 50)

# ==========================
# Find the middle slice
# ==========================

# MRI is stored as (X, Y, Z)

x_middle = data.shape[0] // 2
y_middle = data.shape[1] // 2
z_middle = data.shape[2] // 2

# ==========================
# Display slices
# ==========================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sagittal (left-right)
axes[0].imshow(np.rot90(data[x_middle, :, :]), cmap="gray")
axes[0].set_title("Sagittal")
axes[0].axis("off")

# Coronal (front-back)
axes[1].imshow(np.rot90(data[:, y_middle, :]), cmap="gray")
axes[1].set_title("Coronal")
axes[1].axis("off")

# Axial (top-bottom)
axes[2].imshow(np.rot90(data[:, :, z_middle]), cmap="gray")
axes[2].set_title("Axial")
axes[2].axis("off")

plt.tight_layout()
plt.show()