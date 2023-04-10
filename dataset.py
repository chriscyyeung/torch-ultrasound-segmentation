import random
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class BUSDataset(Dataset):
    def __init__(self, imgs, masks, transform=None, target_transform=None, joint_transform=None):
        self.images = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


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
        mask = mask.astype("float32")
        return F.to_pil_image(img).convert("RGB"), F.to_pil_image(mask).convert("L")
    

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


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
