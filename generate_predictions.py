import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

from dataset import BUSDataset, ToTensor, Compose, ToPILImage
from Models.unet import UNet
from Models.ggnet import GGNet
from utils import *


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_paths = [
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/ggnet_best_model_20230410_223717.pt", 
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/ggnet_best_model_20230411_030455.pt", 
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/unet_best_model_20230411_073221.pt", 
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/unet_best_model_20230411_092018.pt", 
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/ggnet_best_model_20230411_112657.pt", 
        "c:/Users/Chris/Documents/CISC881/torch-ultrasound-segmentation/SavedModels/unet_best_model_20230411_155528.pt"
    ]

    # Create project directories, if they don't exist
    data_arrays_fullpath, results_save_fullpath, models_save_fullpath, logs_save_fullpath \
        = create_standard_project_folders("d:/UltrasoundSegmentation")
    
    # Get test data
    test_img_idx = list(range(28, 32))
    ultrasound_arrays, segmentation_arrays = load_ultrasound_data(data_arrays_fullpath)
    test_ultrasounds = get_data_array(ultrasound_arrays, test_img_idx)
    test_segmentations = get_data_array(segmentation_arrays, test_img_idx)

    # Initialize transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        ToTensor(),
        transforms.Resize((256, 256), antialias=True)
    ])
    joint_transform = Compose([ToPILImage()])

    # Choose 1 image from each patient
    test_indices = [90, 248, 387, 526]
    for i in range(len(test_indices)):
        image = test_ultrasounds[test_indices[i]]
        mask = test_segmentations[test_indices[i]]
        orig_image = Image.fromarray(np.squeeze(image, axis=2)).resize((256, 256), resample=Image.BILINEAR)

        # Prepare image and mask for model
        image, mask = joint_transform(image, mask)
        image = transform(image)
        mask = target_transform(mask)
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        # Generate predictions for each model
        result = np.zeros([0, 256, 256])
        for model_path in model_paths:
            # Load model
            model_str = os.path.basename(model_path).split("_")[0]
            if model_str == "unet":
                model = UNet(3, 64, 1)
            elif model_str == "ggnet":
                model = GGNet(num_classes=2)
            else:
                raise ValueError(f"Could not parse model from saved model: {model_path}")
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            # Generate prediction
            if model_str == "unet":
                output = F.sigmoid(model(image))
            
            elif model_str == "ggnet":
                o0, o1, o2, o3, o4, o5 = model(image)
                output = F.sigmoid(o0)
            result = np.concatenate((result, output.detach().cpu().numpy()[0]), axis=0)
            
        # Generate figure
        fig, axes = plt.subplots(1, 8, figsize=(30, 5))
        axes[0].imshow(np.flipud(orig_image), cmap="gray", aspect="auto")
        axes[1].imshow(np.flipud(mask[0][0].cpu().numpy()), cmap="gray", aspect="auto")
        axes[2].imshow(np.flipud(result[0]), cmap="gray", aspect="auto")
        axes[3].imshow(np.flipud(result[1]), cmap="gray", aspect="auto")
        axes[4].imshow(np.flipud(result[2]), cmap="gray", aspect="auto")
        axes[5].imshow(np.flipud(result[3]), cmap="gray", aspect="auto")
        axes[6].imshow(np.flipud(result[4]), cmap="gray", aspect="auto")
        axes[7].imshow(np.flipud(result[5]), cmap="gray", aspect="auto")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.savefig(f"Examples/example_patient_{test_indices[i]}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
