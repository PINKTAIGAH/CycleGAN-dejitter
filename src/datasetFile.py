from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None):
        self.rootZebra = root_zebra
        self.rootHorse = root_horse
        self.transform = transform

        # Create list of all images in files
        self.zebraImages = os.listdir(root_zebra)
        self.zebraLength = len(self.zebraImages)
        self.horseImages = os.listdir(root_horse)
        self.horseLength = len(self.horseImages)

        self.lengthDataset = max(self.zebraLength, self.horseLength)

    def __len__(self):
        return self.lengthDataset

    def __getitem__(self, idx):
        # Modulo used ot avoid index being larger than list length
        zebraImage = self.zebraImages[idx % self.zebraLength] 
        horseImage = self.horseImages[idx % self.horseLength] 

        zebraPath = os.path.join(self.rootZebra, zebraImage)
        horsePath = os.path.join(self.rootHorse, horseImage)

        zebraImage = np.array(Image.open(zebraPath).convert("RGB"))
        horseImage = np.array(Image.open(horsePath).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=zebraImage, image0=horseImage)
            zebraImage = augmentations["image"]
            horseImage = augmentations["image0"]
        return zebraImage, horseImage
