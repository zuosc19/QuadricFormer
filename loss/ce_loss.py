from .base_loss import BaseLoss
from . import GPD_LOSS
import torch.nn.functional as F
import torch


@GPD_LOSS.register_module()
class CELoss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=255, loss_name=None,
                 cls_weight=None, input_dict=None, use_softmax=True, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        if loss_name is not None:
            self.loss_name = loss_name
        self.loss_func = self.ce_loss
        self.use_softmax = use_softmax
        self.ignore_label = ignore_label
        self.cls_weight = torch.tensor(cls_weight).cuda() if cls_weight is not None else None
        if self.cls_weight is not None:
            num_classes = len(cls_weight)
            self.cls_weight = num_classes * F.normalize(self.cls_weight, p=1, dim=-1)
    
    def ce_loss(self, ce_input, ce_label):
        # input: -1, c
        # output: -1, 1
        ce_input = ce_input.float()
        ce_label = ce_label.long()
        if self.use_softmax:
            ce_loss = F.cross_entropy(ce_input, ce_label, weight=self.cls_weight, 
                                  ignore_index=self.ignore_label)
        else:
            ce_input = torch.clamp(ce_input, 1e-6, 1. - 1e-6)
            ce_loss = F.nll_loss(torch.log(ce_input), ce_label, weight=self.cls_weight,
                                  ignore_index=self.ignore_label)
        return ce_loss


@GPD_LOSS.register_module()
class PixelDistributionLoss(BaseLoss):

    def __init__(
        self,
        weight=1.0,
        use_sigmoid=True,
        input_dict=None
    ):
        
        super().__init__(weight)

        if input_dict is None:
            self.input_dict = {
                'pixel_logits': 'pixel_logits',
                'pixel_gt': 'pixel_gt',
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.loss_voxel
        self.use_sigmoid = use_sigmoid

    def loss_voxel(self, pixel_logits, pixel_gt):
        if self.use_sigmoid:
            pixel_logits = torch.sigmoid(pixel_logits)
        else:
            pixel_logits = torch.softmax(pixel_logits, dim=-1)
        loss = F.binary_cross_entropy(pixel_logits, pixel_gt.float())
        return loss


@GPD_LOSS.register_module()
class BCELoss(BaseLoss):
    
    def __init__(self, weight=1.0, pos_weight=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'ce_input': 'ce_input',
                'ce_label': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.ce_loss
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
    
    def ce_loss(self, ce_input, ce_label):
        # input: -1, 1
        # output: -1, 1
        ce_input = ce_input.float()
        ce_label = ce_label.float()
        ce_loss = F.binary_cross_entropy_with_logits(ce_input, ce_label, weight=self.pos_weight)
        return ce_loss