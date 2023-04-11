import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)

        # Flattened labels and predictions
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)

        intersection = (y_true_f * y_pred_f).sum()
        dice = (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

        return 1 - dice


# Adapted from https://github.com/LIVIAETS/boundary-loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)

        multiplied = einsum("bkwh,bkwh->bkwh", y_pred, y_true)
        loss = multiplied.mean()

        return loss


class DiceBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()

        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, y_pred, y_true):
        return self.alpha * self.dice_loss(y_pred, y_true) \
                + (1 - self.alpha) * self.boundary_loss(y_pred, y_true)


if __name__ == "__main__":
    true = torch.ones((32, 1, 300, 300))
    pred = torch.zeros((32, 1, 300, 300))
    loss = DiceLoss()
    # loss = BoundaryLoss()
    print(loss(pred, true))
