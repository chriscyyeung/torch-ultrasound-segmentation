import datetime
import argparse
import tqdm
import wandb
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import BCEWithLogitsLoss
from torchmetrics.classification import BinaryAccuracy

from dataset import BUSDataset
from loss import DiceLoss
from Models.unet import UNet
from utils import *


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", 
        type=str, 
        required=True
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
    parser.add_argument(
        "--train_split", 
        type=float,  
        default=0.8,
        choices=range(0,1), 
        metavar="[0-1]"
    )
    return parser


def main(FLAGS):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Project folder: {(project_dir := FLAGS.train_img_dirs)}")
    print(f"Epochs:         {(epochs := FLAGS.epochs)}")
    print(f"Batch size:     {(batch_size := FLAGS.batch_size)}")
    print(f"Learning rate:  {(lr := FLAGS.lr)}")

    # Create project directories, if they don't exist
    data_arrays_fullpath, results_save_fullpath, models_save_fullpath, logs_save_fullpath \
        = create_standard_project_folders(project_dir)

    # Load images and masks
    ultrasound_arrays, segmentation_arrays = load_ultrasound_data(data_arrays_fullpath)

    # Split data into training, validation, and test sets (by patient)
    val_test_splits = (1 - FLAGS.train_split) / 2
    train_indices, val_indices, test_indices \
        = get_train_test_val_indices(ultrasound_arrays, FLAGS.train_split, val_test_splits, val_test_splits)
    

    # TODO: Initialize transforms
    transform = transforms.Compose([ToTensor(), 
                                    transforms.RandomHorizontalFlip(0.5)])
    
    target_transform = transforms.Compose([ToTensor(),
                                           transforms.RandomHorizontalFlip(0.5)])

    # Initialize datasets
    train_dataset = BUSDataset(train_img_dirs, 
                                 mask_dir, 
                                 transform=transform, 
                                 target_transform=target_transform,
                                 preprocess=preprocess)
    val_dataset = BUSDataset([val_img_dir], 
                               mask_dir,  
                               transform=transform, 
                               target_transform=target_transform,
                               preprocess=preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    model = UNet(1, 64, 1)
    model.to(device)

    # Training settings
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.90)
    loss_fn = DiceLoss().cuda()
    # loss_fn = BCEWithLogitsLoss(pos_weight=torch.FloatTensor([9.])).cuda()
    metric = BinaryAccuracy().to(device)

    # Initialize logger
    experiment = wandb.init(
        project="cisc881-prostate-cancer-segmentation",
        config={
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "loss_fn": "dice"
        }
    )

    # Training loop
    save_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with tqdm.tqdm(total=epochs) as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            best_val_loss = np.inf
            for epoch_idx, batch in enumerate(tqdm.tqdm(train_dataloader)):
                image, label = batch[0].to(device), batch[1].to(device)  # use gpu
                optimizer.zero_grad()
                output = model(image)
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += metric(F.sigmoid(output), label)
            avg_epoch_loss = epoch_loss / (epoch_idx + 1)
            avg_epoch_acc = epoch_acc / (epoch_idx + 1)
            scheduler.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                for val_idx, batch in enumerate(tqdm.tqdm(val_dataloader)):
                    val_image, val_label = batch[0].to(device), batch[1].to(device)
                    val_output = model(val_image)
                    val_loss += loss_fn(val_output, val_label).item()
                    val_acc += metric(F.sigmoid(val_output), val_label)
                avg_val_loss = val_loss / (val_idx + 1)
                avg_val_acc = val_acc / (val_idx + 1)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 
                            f"TrainedModels/best_{img_type}_model_{save_timestamp}.pt")
                print("Best model saved!")
            
            experiment.log({
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": avg_epoch_loss,
                "val_loss": avg_val_loss,
                "train_acc": avg_epoch_acc,
                "val_acc": avg_val_acc
            })
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}/{epochs}, Epoch Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    FLAGS = get_parser().parse_args()
    main(FLAGS)
