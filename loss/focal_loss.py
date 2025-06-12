# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from .base_loss import BaseLoss
from . import GPD_LOSS
import torch
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        loss = loss * weight
    loss = loss.sum(-1).mean()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@GPD_LOSS.register_module()
class FocalLoss(BaseLoss):

    def __init__(self, weight=1.0, gamma=2.0, alpha=0.25, ignore_label=255,
                 cls_weight=None, cls_freq=None, input_dict=None, **kwargs):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
        """
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'target': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_label = ignore_label
        if cls_weight:
            self.cls_weight = torch.tensor(cls_weight).cuda()
        elif cls_freq:
            self.cls_weight = torch.from_numpy(1 / np.log(cls_freq)).cuda()
        
        H, W = 256, 256       # hard coding
        xy, yx = torch.meshgrid([torch.arange(H)-H/2,  torch.arange(W)-W/2])
        c = torch.stack([xy,yx], 2)
        c = torch.norm(c, 2, -1)
        c_max = c.max()
        self.c = (c/c_max + 1).cuda()
  
        
    def focal_loss(self, pred, target):
        pred = pred.float()
        target = target.long()

        B, H, W, D = target.shape
        # c = self.c[None, :, :, None].repeat(B, 1, 1, D).reshape(-1)
        c = torch.ones_like(target).reshape(-1).cuda()

        visible_mask = (target!=self.ignore_label).reshape(-1).nonzero().squeeze(-1)
        weight_mask = self.cls_weight[None,:] * c[visible_mask, None]
        # visible_mask[:, None]

        num_classes = pred.size(1)
        pred = pred.permute(0, 2, 3, 4, 1).reshape(-1, num_classes)[visible_mask]
        target = target.reshape(-1)[visible_mask]

        loss_cls = sigmoid_focal_loss(
            pred,
            target,
            weight_mask,
            gamma=self.gamma,
            alpha=self.alpha)
        
        return loss_cls