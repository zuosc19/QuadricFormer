import torch.nn as nn
from . import GPD_LOSS

@GPD_LOSS.register_module()
class MultiLoss(nn.Module):

    def __init__(self, loss_cfgs):
        super().__init__()
        self.num_losses = len(loss_cfgs)
        losses = []
        for loss_cfg in loss_cfgs:
            losses.append(GPD_LOSS.build(loss_cfg))
        self.losses = nn.ModuleList(losses)

    def forward(self, inputs):
        loss_dict = {}
        tot_loss = 0.
        for loss_func in self.losses:
            loss = loss_func(inputs)
            tot_loss += loss
            loss_name = getattr(loss_func, 'loss_name', loss_func.__class__.__name__)
            loss_dict.update({
                loss_name: \
                loss.detach().item() / loss_func.weight
            })
        
        return tot_loss, loss_dict