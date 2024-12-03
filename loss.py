import torch
import torch.nn.functional as F


def dice_loss(prediction, target, smooth=1.0):
    
    i_, t_ = prediction.view(-1), target.view(-1)
    inter = (i_ * t_).sum()

    return 1 - ((2. * inter + smooth) / (i_.sum() + t_.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss
