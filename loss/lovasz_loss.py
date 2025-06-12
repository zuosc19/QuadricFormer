from .base_loss import BaseLoss
from . import GPD_LOSS
from utils.lovasz_losses import lovasz_softmax, lovasz_hinge
import torch


@GPD_LOSS.register_module()
class LovaszLoss(BaseLoss):
    
    def __init__(self, weight=1.0, empty_idx=None, ignore_label=None, input_dict=None, use_softmax=True, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.use_softmax = use_softmax
        self.loss_func = self.lovasz_loss
        self.empty_idx = empty_idx
        self.ignore_label = ignore_label
    
    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, c, h, w, z
        # output: -1, h, w, z
        if self.use_softmax:
            lovasz_input = torch.softmax(lovasz_input.float(), dim=1)
        lovasz_label = lovasz_label.long()

        B, C, H, W, D = lovasz_input.size()
        lovasz_input = lovasz_input.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)  # B * H * W * D, C -> P, C
        lovasz_label = lovasz_label.view(-1)    # B * H * W * D
        empty_mask = (lovasz_label == self.empty_idx)
        lovasz_label = lovasz_label[~empty_mask]
        lovasz_input = lovasz_input[~empty_mask]
        lovasz_loss = lovasz_softmax(lovasz_input, lovasz_label, ignore=self.ignore_label)
        return lovasz_loss


@GPD_LOSS.register_module()
class LovaszHingeLoss(BaseLoss):
    
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'lovasz_input': 'lovasz_input',
                'lovasz_label': 'lovasz_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.lovasz_loss
    
    def lovasz_loss(self, lovasz_input, lovasz_label):
        # input: -1, h, w, z
        # output: -1, h, w, z
        lovasz_input = lovasz_input.float()
        lovasz_label = lovasz_label.long()
        lovasz_loss = lovasz_hinge(lovasz_input, lovasz_label)
        return lovasz_loss