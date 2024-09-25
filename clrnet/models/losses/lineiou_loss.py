import torch

##L1-smooth
'''
import torch.nn as nn


def smooth_l1_loss(input, target, beta=1.0 / 99.0):
    # 计算平滑项
    abs_diff = torch.abs(input - target)
    smooth_l1_sign = (abs_diff < beta).float()
    smooth_l1_dist = smooth_l1_sign * 0.5 * torch.pow(input - target, 2) / beta + (1 - smooth_l1_sign) * (
                abs_diff - 0.5 * beta)

    # 返回平滑L1损失
    return smooth_l1_dist.mean()

'''
##L1-smooth



def line_iou(pred, target, img_w, length=5, aligned=True): #15
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)
        union = torch.max(px2, tx2) - torch.min(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (torch.min(px2[:, None, :], tx2[None, ...]) -
               torch.max(px1[:, None, :], tx1[None, ...]))
        union = (torch.max(px2[:, None, :], tx2[None, ...]) -
                 torch.min(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()