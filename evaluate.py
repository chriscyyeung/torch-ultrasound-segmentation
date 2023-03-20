import datetime
import argparse
import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional import dice

from dataset import PICAIDataset, ToTensor
from Models.unet import UNet


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--img_type", type=str, choices={"t2w", "hbv", "adc"}, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test image directories: {(test_img_dir := FLAGS.train_img_dirs)}")
    print(f"Mask directory:         {(mask_dir := FLAGS.mask_dir)}")
    print(f"Image type:             {(img_type := FLAGS.img_type)}")
    print(f"Model path:             {(model_path := FLAGS.model_path)}")
    print(f"Preprocess:             {(preprocess := FLAGS.preprocess)}")
    print(f"Batch size:             {(batch_size := FLAGS.batch_size)}")

    # Initialize dataset
    transform = transforms.Compose([ToTensor()], transforms.Normalize(0, 1))
    target_transform = transforms.Compose([ToTensor()])
    test_dataset = PICAIDataset([test_img_dir],
                                mask_dir,
                                img_type,
                                transform=transform,
                                target_transform=target_transform,
                                preprocess=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize metrics
    accuracy = BinaryAccuracy().to(device)

    # Load model
    model = UNet(1, 64, 1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Testing loop
    test_acc = 0
    test_dice = 0
    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                image, label = batch[0].to(device), batch[1].to(device)
                output = model(image)
                output = torch.sigmoid(output)
                test_acc += accuracy(output, label)
                test_dice += dice(output, label)

            pbar.update(1)
            pbar.set_description(f"Accuracy: {test_acc / (i + 1)}, Dice: {test_dice / (i + 1)}")

    print(f"Average test accuracy: {test_acc / len(test_dataloader)}")
    print(f"Average test dice: {test_dice / len(test_dataloader)}")

    # Save results


if __name__ == "__main__":
    FLAGS = get_parser().parse_args()
    main(FLAGS)
