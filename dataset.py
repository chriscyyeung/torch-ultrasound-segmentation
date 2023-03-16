import os
import glob
import re
import tqdm
import random
import numpy as np
import torch
import torchvision
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


class PICAIDataset(Dataset):
    def __init__(self, img_dirs, mask_dir, img_type, transform=None, target_transform=None, preprocess=False):
        assert img_type in {"t2w", "hbv", "adc"}

        # img_dir expects a list of directories for each fold
        volumes = []
        for i in range(len(img_dirs)):
            volumes += glob.glob(os.path.join(img_dirs[i], f"**/*{img_type}.mha"))

        # Extract image IDs to match with ground truth
        case_ids = [name.split(os.sep)[-1][:5] for name in volumes]

        if img_type == "t2w":  # Prostate gland ground truth
            mask_volumes = glob.glob(os.path.join(mask_dir, "anatomical_delineations/whole_gland/AI/Bosma22b/*.nii.gz"))
        else:  # Cancer lesion ground truth
            mask_volumes = glob.glob(os.path.join(mask_dir, "csPCa_lesion_delineations/AI/Bosma22a/*.nii.gz"))
        mask_volumes = [mask for mask in mask_volumes if mask.split(os.sep)[-1][:5] in case_ids]

        self.images = []
        self.masks = []
        if preprocess:
            # Preprocess volumes and save as image to load later
            print(f"Resampling, cropping, and slicing {img_type} MRI volumes...")
            for volume_name in tqdm.tqdm(volumes):
                self.images += self.preprocess_volume(volume_name, (0.5, 0.5, 3.0), (300, 300, 16), "image")
            
            print("Resampling, cropping, and slicing segmentation volumes...")
            for mask_name in tqdm.tqdm(mask_volumes):
                self.masks += self.preprocess_volume(mask_name, (0.5, 0.5, 3.0), (300, 300, 16), "label") 
        else:  # 2D slices already saved
            for i in range(len(img_dirs)):
                self.images += glob.glob(os.path.join(img_dirs[i], f"**/*{img_type}*.tiff"))
            
            if img_type == "t2w":  # Prostate gland ground truth
                mask_images = glob.glob(os.path.join(mask_dir, "anatomical_delineations/whole_gland/AI/Bosma22b/*.tiff"))
            else:  # Cancer lesion ground truth
                mask_images = glob.glob(os.path.join(mask_dir, "csPCa_lesion_delineations/AI/Bosma22a/*.tiff"))
            self.masks = [mask for mask in mask_images if mask.split(os.sep)[-1][:5] in case_ids]
        
        # Sort images and masks to ensure proper indexing
        self.images.sort(key=self.natural_keys)
        self.masks.sort(key=self.natural_keys)

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
    
    def preprocess_volume(self, volume_dir, spacing, size, type):
        volume = sitk.ReadImage(volume_dir)
        volume = sitk.Cast(volume, sitk.sitkFloat32)
        interpolator = sitk.sitkBSpline if type == "image" else sitk.sitkNearestNeighbor
        volume = self.resample_to_new_spacing(volume, spacing, interpolator)  # Resample to 0.5x0.5x3.0mm

        # Crop to 300x300x16
        center = tuple(dim // 2 for dim in volume.GetSize())
        volume = volume[center[0] - size[0] // 2:center[0] + size[0] // 2,
                        center[1] - size[1] // 2:center[1] + size[1] // 2,
                        center[2] - size[2] // 2:center[2] + size[2] // 2]
        
        images = []
        # Slice the volume and save image
        for slice_idx in range(volume.GetSize()[2]):
            image = volume[:, :, slice_idx]
            image_name = os.path.join(os.path.abspath(os.path.join(volume_dir, os.pardir)), 
                                      volume_dir.split(os.sep)[-1].split(".")[0] + f"_{slice_idx}.tiff")
            sitk.WriteImage(image, image_name)
            images.append(image_name)

        return images

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r"(\d+)", text.split(os.sep)[-1])]
    
    @staticmethod
    def resample_to_new_spacing(image, spacing, interpolator):
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(np.round(old_size * (old_spacing / new_spacing))) 
                    for old_size, old_spacing, new_spacing in zip(original_size, original_spacing, spacing)]
        
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetInterpolator(interpolator)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputSpacing(spacing)
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(image.GetPixelIDValue())
        image = resample.Execute(image)

        return image

    @staticmethod
    def atoi(text):
        return int(text) if text.isdigit() else text


# PyTorch ToTensor() normalizes from [0, 255] to [0, 1]
class ToTensor(object):
    def __call__(self, image):
        image = image[..., np.newaxis]
        image = image.transpose((2, 0, 1))  # Swap axis from (H, W, C) to (C, H, W)
        image = torch.from_numpy(image).contiguous()

        return image


if __name__ == "__main__":
    train_img_dirs = ["e:/CISC881/picai_public_images_fold1", 
                      "e:/CISC881/picai_public_images_fold2", 
                      "e:/CISC881/picai_public_images_fold4"]
    val_img_dirs = ["e:/CISC881/picai_public_images_fold3"]
    test_img_dirs = ["e:/CISC881/picai_public_images_fold0"]
    mask_dir = "e:/CISC881/picai_labels"

    transform = transforms.Compose([ToTensor(), 
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.Normalize(0, 1)])
    
    target_transform = transforms.Compose([ToTensor(),
                                           transforms.RandomHorizontalFlip(0.5)])

    t2w_data = PICAIDataset(train_img_dirs, mask_dir, "t2w", transform=transform, target_transform=target_transform)
    t2w_val = PICAIDataset(val_img_dirs, mask_dir, "t2w", transform=transform, target_transform=target_transform)
    t2w_test = PICAIDataset(test_img_dirs, mask_dir, "t2w", transform=transform, target_transform=target_transform)

    hbv_data = PICAIDataset(train_img_dirs, mask_dir, "hbv", transform=transform, target_transform=target_transform)
    hbv_val = PICAIDataset(val_img_dirs, mask_dir, "hbv", transform=transform, target_transform=target_transform)
    hbv_test = PICAIDataset(test_img_dirs, mask_dir, "hbv", transform=transform, target_transform=target_transform)

    adc_data = PICAIDataset(train_img_dirs, mask_dir, "adc", transform=transform, target_transform=target_transform)
    adc_val = PICAIDataset(val_img_dirs, mask_dir, "adc", transform=transform, target_transform=target_transform)
    adc_test = PICAIDataset(test_img_dirs, mask_dir, "adc", transform=transform, target_transform=target_transform)

    print(f"t2w training images: {len(t2w_data)}, t2w training masks: {len(t2w_data.masks)}")
    print(f"t2w validation images: {len(t2w_val)}, t2w validation masks: {len(t2w_val.masks)}")
    print(f"t2w test images: {len(t2w_test)}, hbv test masks: {len(t2w_test.masks)}")

    print(f"hbv training images: {len(hbv_data)}, hbv training masks: {len(hbv_data.masks)}")
    print(f"hbv validation images: {len(hbv_val)}, hbv validation masks: {len(hbv_val.masks)}")
    print(f"hbv test images: {len(hbv_test)}, hbv test masks: {len(hbv_test.masks)}")

    print(f"adc training images: {len(adc_data)}, adc training masks: {len(adc_data.masks)}")
    print(f"adc validation images: {len(adc_val)}, adc validation masks: {len(adc_val.masks)}")
    print(f"adc test images: {len(adc_test)}, adc test masks: {len(adc_test.masks)}")

    sample = hbv_data[1118]
    plt.imshow(sample[0][0], cmap="gray")
    plt.show()

    plt.imshow(sample[1][0], cmap="gray")
    plt.show()
