from torch.utils.data import DataLoader

from data import data 


dataset = BrainMRIDataset("data/processed")

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for images, names in loader:

    print(images.shape)
    print(names)

    break