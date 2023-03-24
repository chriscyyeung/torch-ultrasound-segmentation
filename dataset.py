import os
import glob
import re
import tqdm
import random
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class BUSDataset(Dataset):
    def __init__(self, img_dirs, mask_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        label = sitk.ReadImage(self.masks[idx])

        image = sitk.GetArrayFromImage(image).astype(np.float32)
        label = sitk.GetArrayFromImage(label).astype(np.float32)

        # Seed to allow image and label to have same random transformations
        seed = np.random.randint(2023)

        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform:
            image = self.transform(image)
        
        random.seed(seed)
        torch.manual_seed(seed)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
