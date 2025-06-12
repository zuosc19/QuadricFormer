from .base_loss import BaseLoss
from . import GPD_LOSS
import torch.nn.functional as F
import torch

@GPD_LOSS.register_module()
class Geo_Scal_Loss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=255,
                 empty_idx=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'ssc_target': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.geo_scal_loss
        self.ignore_label = ignore_label
        self.empty_idx = empty_idx
    
    def geo_scal_loss(self, pred, ssc_target):
        pred = pred.float()
        ssc_target = ssc_target.long()

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, self.empty_idx]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != self.ignore_label
        nonempty_target = ssc_target != self.empty_idx
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        eps = 1e-5
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / (nonempty_probs.sum()+eps)
        recall = intersection / (nonempty_target.sum()+eps)
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / ((1 - nonempty_target).sum()+eps)
        return (
            F.binary_cross_entropy(precision, torch.ones_like(precision))
            + F.binary_cross_entropy(recall, torch.ones_like(recall))
            + F.binary_cross_entropy(spec, torch.ones_like(spec))
        )


@GPD_LOSS.register_module()
class Sem_Scal_Loss(BaseLoss):
    
    def __init__(self, weight=1.0, ignore_label=255,
                 sem_cls_range=None, input_dict=None, **kwargs):
        super().__init__(weight)
        
        if input_dict is None:
            self.input_dict = {
                'pred': 'ce_input',
                'ssc_target': 'ce_label'
            }
        else:
            self.input_dict = input_dict
        self.loss_func = self.sem_scal_loss
        self.ignore_label = ignore_label
        self.sem_cls_range = sem_cls_range

    def sem_scal_loss(self, pred, ssc_target):
        pred = pred.float()
        ssc_target = ssc_target.long()

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)
        loss = 0
        count = 0
        mask = ssc_target != self.ignore_label
        n_classes = pred.shape[1]
        for i in range(self.sem_cls_range[0], self.sem_cls_range[1]):

            # Get probability of class i
            p = pred[:, i]

            # Remove unknown voxels
            target_ori = ssc_target
            p = p[mask]
            target = ssc_target[mask]

            completion_target = torch.ones_like(target)
            completion_target[target != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                if torch.sum(p) > 0:
                    precision = nominator / (torch.sum(p))
                    loss_precision = F.binary_cross_entropy(
                        precision, torch.ones_like(precision)
                    )
                    loss_class += loss_precision
                if torch.sum(completion_target) > 0:
                    recall = nominator / (torch.sum(completion_target))
                    loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                    loss_class += loss_recall
                if torch.sum(1 - completion_target) > 0:
                    specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                        torch.sum(1 - completion_target)
                    )
                    loss_specificity = F.binary_cross_entropy(
                        specificity, torch.ones_like(specificity)
                    )
                    loss_class += loss_specificity
                loss += loss_class
        return loss / count