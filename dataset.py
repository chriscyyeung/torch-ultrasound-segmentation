import os
import glob
import tqdm
import torch
import torchvision
import SimpleITK as sitk
from torch.utils.data import Dataset


class PICAIDataset(Dataset):
    def __init__(self, img_dirs, mask_dir, img_type, transform=None):
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

        # Preprocess volumes and save as image to load later
        self.images = []
        print(f"Resampling, cropping, and slicing {img_type} MRI volumes...")
        for volume_name in tqdm.tqdm(volumes):
            self.images += self.preprocess_volume(volume_name, (0.5, 0.5, 3.0), (300, 300, 16))
        
        self.masks = []
        print("Resampling, cropping, and slicing segmentation volumes...")
        for mask_name in tqdm.tqdm(mask_volumes):
            self.masks += self.preprocess_volume(mask_name, (0.5, 0.5, 3.0), (300, 300, 16))

        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        label = sitk.ReadImage(self.masks[idx])

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def preprocess_volume(self, volume_dir, spacing, size):
        volume = sitk.ReadImage(volume_dir)
        volume = self.resample_to_new_spacing(volume, spacing)  # Resample to 0.5x0.5x3.0mm
        # Crop to 300x300x16
        center = tuple(dim // 2 for dim in volume.GetSize())
        volume = volume[center[0] - size[0] // 2:center[0] + size[0] // 2,
                        center[1] - size[1] // 2:center[1] + size[1] // 2,
                        center[2] - size[2] // 2:center[2] + size[2] // 2]
        images = []
        for slice_idx in range(volume.GetSize()[2]):
            image = volume[:, :, slice_idx]
            image_name = os.path.join(os.path.abspath(os.path.join(volume_dir, os.pardir)), 
                                      volume_dir.split(os.sep)[-1].split(".")[0] + f"_{slice_idx}.tiff")
            sitk.WriteImage(image, image_name)
            images.append(image_name)
        return images
    
    @staticmethod
    def resample_to_new_spacing(image, spacing, interpolator=sitk.sitkBSpline):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator = interpolator
        resample.SetOutputDirection = image.GetDirection()
        resample.SetOutputOrigin = image.GetOrigin()
        resample.SetOutputSpacing(spacing)
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_size = [int(round(old_size * old_spacing / new_spacing)) 
                    for old_size, old_spacing, new_spacing in zip(original_size, original_spacing, spacing)]
        resample.SetSize(new_size)
        image = resample.Execute(image)
        return image


if __name__ == "__main__":
    train_img_dirs = ["e:/CISC881/picai_public_images_fold1", 
                      "e:/CISC881/picai_public_images_fold2", 
                      "e:/CISC881/picai_public_images_fold4"]
    val_img_dirs = ["e:/CISC881/picai_public_images_fold3"]
    test_img_dirs = ["e:/CISC881/picai_public_images_fold0"]
    mask_dir = "e:/CISC881/picai_labels"

    # t2w_data = PICAIDataset(train_img_dirs, mask_dir, "t2w")
    # hbv_data = PICAIDataset(train_img_dirs, mask_dir, "hbv")
    # adc_data = PICAIDataset(train_img_dirs, mask_dir, "adc")

    # print(f"t2w training images: {len(t2w_data)}, t2w training masks: {len(t2w_data.masks)}")
    # print(f"hbv training images: {len(hbv_data)}, hbv training masks: {len(hbv_data.masks)}")
    # print(f"adc training images: {len(adc_data)}, adc training masks: {len(adc_data.masks)}")

    t2w_val = PICAIDataset(val_img_dirs, mask_dir, "t2w")
    print(f"t2w validation images: {len(t2w_val)}, t2w validation masks: {len(t2w_val.masks)}")
    print(t2w_val[0][0].shape, t2w_val[0][1].shape)

    hbv_test = PICAIDataset(test_img_dirs, mask_dir, "hbv")
    print(f"hbv test images: {len(hbv_test)}, hbv test masks: {len(hbv_test.masks)}")
    print(hbv_test[0][0].shape, hbv_test[0][1].shape)
