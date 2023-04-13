import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset


class BUSDataset(Dataset):
    def __init__(self, imgs, masks, transform=None, target_transform=None, 
                 joint_transform=None, dist_map_transform=None):
        self.images = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.dist_map_transform = dist_map_transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        dist_map = None

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        if self.dist_map_transform:
            dist_map = self.dist_map_transform(mask)

        return image, mask, dist_map


class ToTensor(object):
    def __call__(self, img):
        return F.pil_to_tensor(img)


class OneHotEncode(object):
    def __call__(self, img):
        return F.one_hot(img, num_classes=2)


# Joint transforms taken from https://github.com/xorangecheng/GlobalGuidance-Net/blob/main/datasets/joint_transforms.py
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
    

class ToPILImage(object):
    def __call__(self, img, mask):
        img = img.astype("float32")
        mask = mask.astype("uint8")
        return F.to_pil_image(img).convert("RGB"), F.to_pil_image(mask).convert("L")
    

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BICUBIC), mask.resize(self.size, Image.NEAREST)


class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)
    

# Distance map transforms
class OneHotToDistanceMap(object):
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, img):
        res = np.zeros_like(img)
        for k in range(2):  # 2 classes
            posmask = img[k].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                res[k] = distance_transform_edt(negmask, sampling=self.resolution) * negmask \
                    - (distance_transform_edt(posmask, sampling=self.resolution) - 1) * posmask
        return torch.from_numpy(res, dtype=np.uint8)
