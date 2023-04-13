import datetime
import argparse
import json
import tqdm
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, JaccardIndex, Precision, Recall, Dice

from dataset import BUSDataset, ToTensor, Compose, ToPILImage
from Models.unet import UNet
from Models.ggnet import GGNet
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_json", type=str, default="results.json")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Project folder:        {(project_dir := FLAGS.project_dir)}")
    print(f"Model path:            {(model_path := FLAGS.model_path)}")
    print(f"Results JSON filename: {(save_json := FLAGS.save_json)}")
    print(f"Batch size:            {(batch_size := FLAGS.batch_size)}")

    # Create project directories, if they don't exist
    data_arrays_fullpath, results_save_fullpath, models_save_fullpath, logs_save_fullpath \
        = create_standard_project_folders(project_dir)

    # For saving test configuration
    config = {
        "date": datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
        "project_folder": project_dir,
        "model_path": model_path,
        "batch_size": batch_size
    }

    # Load data
    test_img_idx = list(range(28, 33))
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

    # Create dataloader
    test_dataset = BUSDataset(
        test_ultrasounds,
        test_segmentations,
        transform=transform,
        target_transform=target_transform,
        joint_transform=joint_transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize metrics
    acc = BinaryAccuracy().to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    dice = Dice().to(device)
    jaccard = JaccardIndex(task="binary").to(device)

    # Load model
    model_str = os.path.basename(model_path).split("_")[0]
    if model_str == "unet":
        model = UNet(3, 64, 1)
    elif model_str == "ggnet":
        model = GGNet()
    else:
        raise ValueError(f"Could not parse model from saved model: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Testing loop
    test_acc = 0
    test_precision = 0
    test_recall = 0
    test_dice = 0
    test_jaccard = 0
    with tqdm.tqdm(total=len(test_dataloader)) as pbar:
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                image, label, _ = batch[0].to(device), batch[1].to(device), batch[2]

                if model_str == "unet":
                    output = F.sigmoid(model(image))
                
                elif model_str == "ggnet":
                    o0, o1, o2, o3, o4, o5 = model(image)
                    output = F.sigmoid(o0)

                test_acc += acc(output, label)
                test_precision += precision(output, label)
                test_recall += recall(output, label)
                test_dice += dice(output, label)
                test_jaccard += jaccard(output, label)

            pbar.update(1)

    avg_test_acc = test_acc / len(test_dataloader)
    avg_test_precision = test_precision / len(test_dataloader)
    avg_test_recall = test_recall / len(test_dataloader)
    avg_test_dice = test_dice / len(test_dataloader)
    avg_test_jaccard = test_jaccard / len(test_dataloader)

    print(f"Average test accuracy: {avg_test_acc}")
    print(f"Average test precision: {avg_test_precision}")
    print(f"Average test recall: {avg_test_recall}")
    print(f"Average test dice: {avg_test_dice}")
    print(f"Average test jaccard: {avg_test_jaccard}")

    # Save results
    config["accuracy"] = avg_test_acc.item()
    config["precision"] = avg_test_precision.item()
    config["recall"] = avg_test_recall.item()
    config["dice"] = avg_test_dice.item()
    config["jaccard"] = avg_test_jaccard.item()
    
    json_string = json.dumps(config, indent=4)
    with open(save_json, "a+") as f:
        f.write(json_string)


if __name__ == "__main__":
    FLAGS = get_parser().parse_args()
    main(FLAGS)
