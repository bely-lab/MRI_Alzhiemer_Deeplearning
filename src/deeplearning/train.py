"""
train.py
MRI -> Dataset -> DataLoader -> 3D CNN -> Prediction
"""

import torch
from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from model import BrainAgeCNN


def main():

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cpu")

    print(f"Using device: {device}")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = BrainMRIDataset("data/processed")

    print(f"Dataset size: {len(dataset)}")

    # -----------------------------
    # DataLoader
    # -----------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = BrainAgeCNN()
    model.to(device)

    # -----------------------------
    # Load one batch
    # -----------------------------
    images, subjects = next(iter(dataloader))

    images = images.to(device)

    # -----------------------------
    # Forward pass
    # -----------------------------
    predictions = model(images)

    print("\nBatch information")
    print("-" * 40)

    print("Subjects:")
    print(list(subjects))

    print("\nInput shape:")
    print(images.shape)

    print("\nOutput shape:")
    print(predictions.shape)

    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()