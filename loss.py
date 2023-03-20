import torch
import torch.nn as nn
import torch.nn.functional as F


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


if __name__ == "__main__":
    true = torch.ones((1, 300, 300))
    pred = torch.zeros((1, 300, 300))
    loss = DiceLoss()
    print(loss(pred, true))
