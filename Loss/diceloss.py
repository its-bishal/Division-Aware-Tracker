
import torch as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """measure the overlap between predicted masks and ground-truth regions. 
    1 - 2 * Intersection / Union"""
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true):
        pred_probs = F.sigmoid(y_pred)
        y_true_f = y_true.view(-1)
        y_pred_f = pred_probs.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        return (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def forward(self, y_pred, y_true, **kwargs):
        return 1 - self.dice_coef(y_pred, y_true)
    
