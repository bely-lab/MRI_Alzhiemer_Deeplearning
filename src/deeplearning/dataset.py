"""
dataset.py

PyTorch Dataset for loading preprocessed MRI scans.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

def resize_volume(volume, target_size=(128, 128, 128)):
    """
    Resize a 3D MRI volume to a fixed size.
    """

    factors = (
        target_size[0] / volume.shape[0],
        target_size[1] / volume.shape[1],
        target_size[2] / volume.shape[2],
    )

    return zoom(volume, zoom=factors, order=1)
class BrainMRIDataset(Dataset):

    def __init__(self, data_dir):

        self.files = sorted(Path(data_dir).glob("*.nii.gz"))

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]

        image = nib.load(str(file)).get_fdata()

        image = image.astype(np.float32)

        image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image)

        subject = file.stem.replace(".nii", "")

        return image, subject


if __name__ == "__main__":

    dataset = BrainMRIDataset("data/processed")

    print("Dataset size:", len(dataset))

    image, subject = dataset[0]

    print("Subject:", subject)

    print("Tensor shape:", image.shape)