import datetime
import argparse
import tqdm
import wandb
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchmetrics.classification import BinaryAccuracy

from dataset import BUSDataset, ToTensor, OneHotEncode, OneHotToDistanceMap
from dataset import Compose, ToPILImage, FreeScale, RandomHorizontalFlip, RandomRotate
from loss import *
from Models.unet import UNet
from Models.ggnet import GGNet
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--model", 
        type=str,
        default="unet",
        choices=["unet", "ggnet"]
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="dice",
        choices=["dice", "boundary", "db"]
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3
    )
    return parser


def main(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print(f"Project folder: {(project_dir := FLAGS.project_dir)}")
    print(f"Model:          {(model_str := FLAGS.model)}")
    print(f"Loss function:  {(loss_fn_str := FLAGS.loss_fn)}")
    print(f"Epochs:         {(epochs := FLAGS.epochs)}")
    print(f"Batch size:     {(batch_size := FLAGS.batch_size)}")
    print(f"Learning rate:  {(lr := FLAGS.lr)}")

    # Create project directories, if they don't exist
    data_arrays_fullpath, results_save_fullpath, models_save_fullpath, logs_save_fullpath \
        = create_standard_project_folders(project_dir)

    # Load images and masks
    train_img_idx = list(range(0, 24))
    val_img_idx = list(range(24, 28))
    ultrasound_arrays, segmentation_arrays = load_ultrasound_data(data_arrays_fullpath)
    train_ultrasounds = get_data_array(ultrasound_arrays, train_img_idx)
    train_segmentations = get_data_array(segmentation_arrays, train_img_idx)
    val_ultrasounds = get_data_array(ultrasound_arrays, val_img_idx)
    val_segmentations = get_data_array(segmentation_arrays, val_img_idx)

    # Initialize transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    target_transform = transforms.Compose([
        ToTensor(), 
        OneHotEncode()
    ])
    joint_transform = Compose([
        ToPILImage(),
        FreeScale((256, 256)),
        RandomHorizontalFlip(),
        RandomRotate(10)
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_target_transform = transforms.Compose([
        ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        OneHotEncode()
    ])
    val_joint_transform = Compose([ToPILImage()])
    dist_map_transform = transforms.Compose([OneHotToDistanceMap([1, 1])])

    # Initialize datasets
    train_dataset = BUSDataset(
        train_ultrasounds, 
        train_segmentations, 
        transform=transform, 
        target_transform=target_transform,
        joint_transform=joint_transform, 
        dist_map_transform=dist_map_transform
    )
    val_dataset = BUSDataset(
        val_ultrasounds, 
        val_segmentations,
        transform=val_transform,
        target_transform=val_target_transform,
        joint_transform=val_joint_transform, 
        dist_map_transform=dist_map_transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    if model_str == "unet":
        model = UNet(3, 64, 2)
    elif model_str == "ggnet":
        model = GGNet(num_classes=2)
    else:
        raise NotImplementedError
    model.to(device)

    # Training settings
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if loss_fn_str == "dice":
        loss_fn = DiceLoss().cuda()
    elif loss_fn_str == "boundary":
        loss_fn = BoundaryLoss().cuda()
    elif loss_fn_str == "db":
        loss_fn = DiceBoundaryLoss(alpha=0.1).cuda()
    else:
        raise NotImplementedError
    metric = BinaryAccuracy().to(device)

    # Initialize logger
    experiment = wandb.init(
        project="cisc881-breast-ultrasound-segmentation",
        config={
            "model": model_str, "loss_fn": loss_fn_str, "epochs": epochs, "batch_size": batch_size, "lr": lr
        }
    )

    # Training loop
    save_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = np.inf
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            for epoch_idx, batch in enumerate(train_dataloader):
                image, label, dist_map = batch[0].to(device), batch[1].to(device), batch[2].to(device)  # use gpu
                optimizer.zero_grad()

                if model_str == "unet":
                    output = model(image)
                    if loss_fn_str == "dice":
                        loss = loss_fn(output, label)
                    elif loss_fn_str == "boundary":
                        loss = loss_fn(output, dist_map)
                    else:
                        loss = loss_fn(output, label, dist_map)
                    acc = metric(F.sigmoid(output), label)

                elif model_str == "ggnet":
                    o0, o1, o2, o3, o4, o5 = model(image)

                    if loss_fn_str == "dice":
                        loss0 = loss_fn(o0, label)
                        # loss1 = loss_fn(o1, label)
                        # loss2 = loss_fn(o2, label)
                        # loss3 = loss_fn(o3, label)
                        # loss4 = loss_fn(o4, label)
                        # loss5 = loss_fn(o5, label)
                    elif loss_fn_str == "boundary":
                        loss0 = loss_fn(o0, dist_map)
                        # loss1 = loss_fn(o1, dist_map)
                        # loss2 = loss_fn(o2, dist_map)
                        # loss3 = loss_fn(o3, dist_map)
                        # loss4 = loss_fn(o4, dist_map)
                        # loss5 = loss_fn(o5, dist_map)
                    else:
                        loss0 = loss_fn(o0, label, dist_map)
                        # loss1 = loss_fn(o1, label, dist_map)
                        # loss2 = loss_fn(o2, label, dist_map)
                        # loss3 = loss_fn(o3, label, dist_map)
                        # loss4 = loss_fn(o4, label, dist_map)
                        # loss5 = loss_fn(o5, label, dist_map)

                    # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
                    loss = loss0
                    acc = metric(F.sigmoid(o0), label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc

                pbar.update(1)
                pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc:.4f}")

        avg_epoch_loss = epoch_loss / (epoch_idx + 1)
        avg_epoch_acc = epoch_acc / (epoch_idx + 1)

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            with tqdm.tqdm(total=len(val_dataloader)) as pbar:
                for val_idx, batch in enumerate(val_dataloader):
                    val_image, val_label, val_dist_map = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                    if model_str == "unet":
                        val_output = model(val_image)
                        acc = metric(F.sigmoid(val_output), val_label)
                    
                    elif model_str == "ggnet":
                        val_output, val_o1, val_o2, val_o3, val_o4, val_o5 = model(val_image)
                        acc = metric(F.sigmoid(val_output), val_label)
                    
                    if loss_fn_str == "dice":
                        loss = loss_fn(val_output, val_label)
                    elif loss_fn_str == "boundary":
                        loss = loss_fn(val_output, val_dist_map)
                    else:
                        loss = loss_fn(val_output, val_label, val_dist_map)
                    
                    val_loss += loss.item()
                    val_acc += acc

                    pbar.update(1)
                    pbar.set_description(f"loss: {loss.item():.4f}, acc: {acc:.4f}")
                    
            avg_val_loss = val_loss / (val_idx + 1)
            avg_val_acc = val_acc / (val_idx + 1)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(f"SavedModels/{model_str}_best_model_{save_timestamp}.pt")
            torch.save(model.state_dict(), model_save_path)
            print("Best model saved!")
        
        experiment.log({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": avg_epoch_loss,
            "val_loss": avg_val_loss,
            "train_acc": avg_epoch_acc,
            "val_acc": avg_val_acc
        })


if __name__ == "__main__":
    FLAGS = get_parser().parse_args()
    main(FLAGS)
