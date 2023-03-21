import torch
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
from Models.unet import UNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet(1, 64, 1)
    # model.load_state_dict(torch.load("c:/Users/Chris/Documents/CISC881/prostate-cancer-segmentation/TrainedModels/best_t2w_model_20230320_212039.pt"))
    # model.load_state_dict(torch.load("c:/Users/Chris/Documents/CISC881/prostate-cancer-segmentation/TrainedModels/best_hbv_model_20230320_232634.pt"))
    model.load_state_dict(torch.load("c:/Users/Chris/Documents/CISC881/prostate-cancer-segmentation/TrainedModels/best_adc_model_20230321_012953.pt"))
    model.to(device)
    model.eval()

    # Make prediction on image (10522_1000532)
    image_t2w = "d:/Chris/PICAI/data/picai_public_images_fold0/10522/10522_1000532_t2w_8.tiff"
    image_hbv = "d:/Chris/PICAI/data/picai_public_images_fold0/10522/10522_1000532_hbv_8.tiff"
    image_adc = "d:/Chris/PICAI/data/picai_public_images_fold0/10522/10522_1000532_adc_8.tiff"
    mask_t2w = "d:/Chris/PICAI/data/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/10522_1000532_8.tiff"
    mask_hbv = "d:/Chris/PICAI/data/picai_labels/csPCa_lesion_delineations/AI/Bosma22a/10522_1000532_8.tiff"
    mask_adc = "d:/Chris/PICAI/data/picai_labels/csPCa_lesion_delineations/AI/Bosma22a/10522_1000532_8.tiff"

    # Load image
    # image = sitk.ReadImage(image_t2w)
    # image = sitk.ReadImage(image_hbv)
    image = sitk.ReadImage(image_adc)
    # mask = sitk.ReadImage(mask_t2w)
    # mask = sitk.ReadImage(mask_hbv)
    mask = sitk.ReadImage(mask_adc)
    image_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)

    # Get prediction
    image_tensor = torch.from_numpy(image_arr.astype("float32")).unsqueeze(0).unsqueeze(0).to(device)
    pred = F.sigmoid(model(image_tensor))

    # Plot image and mask
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pred.cpu().detach().numpy()[0][0])
    ax[1].imshow(mask_arr)
    plt.show()


if __name__ == "__main__":
    main()
