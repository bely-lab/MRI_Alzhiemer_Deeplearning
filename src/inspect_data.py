"""
inspect_dataset.py

Purpose:
--------
Inspect all MRI subjects in the OASIS dataset and collect
basic information about the scans.

Author: Belaynesh Kndie
"""

from pathlib import Path
import nibabel as nib
import pandas as pd


# ==========================================================
# Dataset location
# ==========================================================

DATASET_PATH = Path("data/raw/OASIS")


# ==========================================================
# Store information from all subjects
# ==========================================================

records = []


# ==========================================================
# Loop through subject folders
# ==========================================================

for subject_dir in sorted(DATASET_PATH.glob("OAS1_*")):

    try:
        # Find MRI header files in RAW folder
        hdr_files = list(subject_dir.glob("RAW/*.hdr"))

        if not hdr_files:
            print(f"No MRI found: {subject_dir.name}")
            continue

        # Use first MRI acquisition
        mri_path = hdr_files[0]

        image = nib.load(str(mri_path))

        shape = image.shape
        voxel_size = image.header.get_zooms()

        records.append(
            {
                "subject": subject_dir.name,
                "file": mri_path.name,
                "shape": shape,
                "voxel_size": voxel_size,
            }
        )

    except Exception as e:
        print(f"Error loading {subject_dir.name}: {e}")


# ==========================================================
# Create DataFrame
# ==========================================================

df = pd.DataFrame(records)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Number of subjects: {len(df)}")

print("\nUnique image shapes:")
print(df["shape"].value_counts())

print("\nUnique voxel sizes:")
print(df["voxel_size"].value_counts())

print("\nFirst 5 subjects:")
print(df.head())

# ==========================================================
# Save report
# ==========================================================

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "dataset_summary.csv"

df.to_csv(output_file, index=False)

print("\nSaved report:")
print(output_file)