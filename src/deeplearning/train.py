"""
train.py

Test the complete deep learning pipeline:
Dataset -> DataLoader -> Model -> Forward Pass
"""

import torch
from torch.utils.data import DataLoader

from dataset import BrainMRIDataset
from model import BrainAgeCNN


def main():

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = BrainMRIDataset("data/processed")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    print(f"Dataset size: {len(dataset)}")

    # -----------------------------
    # Model
    # -----------------------------
    model = BrainAgeCNN()

    model.to(device)

    print(model)

    # -----------------------------
    # One Forward Pass
    # -----------------------------
    images, subjects = next(iter(dataloader))

    images = images.to(device)

    predictions = model(images)

    print("\nSubjects:")
    print(list(subjects))

    print("\nInput shape:")
    print(images.shape)

    print("\nPrediction shape:")
    print(predictions.shape)

    print("\nPredictions:")
    print(predictions)


if __name__ == "__main__":
    main()


    