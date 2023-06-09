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
        # y_pred = F.one_hot(y_pred.long(), num_classes=2).transpose(1, 4).squeeze(-1)

        # Flattened labels and predictions
        y_true_f = y_true.contiguous().view(-1)
        y_pred_f = y_pred.contiguous().view(-1)

        intersection = (y_true_f * y_pred_f).sum()
        dice = (2. * intersection + self.smooth) / (y_true_f.sum() + y_pred_f.sum() + self.smooth)

        return 1 - dice


# Adapted from https://github.com/LIVIAETS/boundary-loss
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, dist_map):
        y_pred = F.softmax(y_pred, dim=1)

        multiplied = einsum("bkwh,bkwh->bkwh", y_pred, dist_map)
        loss = multiplied.mean()

        return loss


class DiceBoundaryLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()

        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, y_pred, y_true, dist_map):
        return self.alpha * self.dice_loss(y_pred, y_true) \
                + (1 - self.alpha) * self.boundary_loss(y_pred, dist_map)


if __name__ == "__main__":
    true = torch.ones((32, 1, 300, 300))
    pred = torch.zeros((32, 1, 300, 300))
    loss = DiceLoss()
    # loss = BoundaryLoss()
    print(loss(pred, true))
