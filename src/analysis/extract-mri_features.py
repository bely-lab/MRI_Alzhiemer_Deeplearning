"""
extract_mri_features.py

Extract quantitative features from all processed MRIs.

Author: Belaynesh Kndie
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


DATASET = Path("data/processed")

records = []

print("Extracting MRI features...\n")

for mri_file in sorted(DATASET.glob("*.nii.gz")):

    print(f"Processing {mri_file.name}")

    img = nib.load(str(mri_file))
    data = img.get_fdata()

    shape = data.shape

    mask = data != 0

    brain = data[mask]

    records.append({

        "subject": mri_file.stem.replace(".nii", ""),

        "shape_x": shape[0],
        "shape_y": shape[1],
        "shape_z": shape[2],

        "brain_voxels": brain.size,

        "mean": brain.mean(),
        "std": brain.std(),

        "min": brain.min(),
        "max": brain.max(),

        "median": np.median(brain),

        "percentile_25": np.percentile(brain,25),
        "percentile_75": np.percentile(brain,75),

    })

df = pd.DataFrame(records)

output = Path("data/mri_features.csv")

df.to_csv(output,index=False)

print("\nDone!")

print(df.head())

print(f"\nSaved to {output}")