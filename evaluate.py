import datetime
import argparse
import json
import tqdm
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional import dice

from dataset import BUSDataset
from Models.unet import UNet


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_img_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_json", type=str, default="results.json")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Test image directories: {(test_img_dir := FLAGS.test_img_dir)}")
    print(f"Mask directory:         {(mask_dir := FLAGS.mask_dir)}")
    print(f"Model path:             {(model_path := FLAGS.model_path)}")
    print(f"Results JSON filename:  {(save_json := FLAGS.save_json)}")
    print(f"Preprocess:             {(preprocess := FLAGS.preprocess)}")
    print(f"Batch size:             {(batch_size := FLAGS.batch_size)}")

    # For saving test configuration
    config = {
        "date": datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        "test_img_dir": test_img_dir,
        "mask_dir": mask_dir,
        "model_path": model_path,
        "batch_size": batch_size
    }

    test_img_idx = list(range(28, 33))

    # TODO: Initialize dataset
    transform = transforms.Compose([transforms.Normalize(0, 1)])
    target_transform = transforms.Compose([])
    test_dataset = BUSDataset([test_img_dir],
                                mask_dir,
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
                test_dice += dice(output, label.long())

            pbar.update(1)
            pbar.set_description(f"Accuracy: {test_acc / (i + 1)}, Dice: {test_dice / (i + 1)}")

    avg_test_acc = test_acc / len(test_dataloader)
    avg_test_dice = test_dice / len(test_dataloader)
    print(f"Average test accuracy: {avg_test_acc}")
    print(f"Average test dice: {avg_test_dice}")

    # Save results
    config["accuracy"] = avg_test_acc.item()
    config["dice"] = avg_test_dice.item()
    json_string = json.dumps(config, indent=4)
    with open(save_json, "a+") as f:
        f.write(json_string)


if __name__ == "__main__":
    FLAGS = get_parser().parse_args()
    main(FLAGS)
