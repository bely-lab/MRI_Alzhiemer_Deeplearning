"""
test_dataloader.py
"""

from torch.utils.data import DataLoader

from dataset import BrainMRIDataset


def main():

    dataset = BrainMRIDataset("data/processed")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    print(f"Dataset size: {len(dataset)}")

    images, subjects = next(iter(dataloader))

    print("\nBatch loaded successfully!")

    print(f"Images shape: {images.shape}")
    print(f"Subjects: {list(subjects)}")


if __name__ == "__main__":
    main()